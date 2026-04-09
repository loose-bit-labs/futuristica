import argparse
import copy
import json
import glob
import logging
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

from model import Neuralistica


class MetaTrainer:
    """Reptile meta-learner for SIREN weight initialisation.

    Trains a starting point theta* such that for any image, K steps of
    gradient descent from theta* reaches a better solution than Sitzmann
    random init.  The result is a weights.npz drop-in for --checkpoint.

    Algorithm (Reptile, Nichol et al. 2018):
        theta = sitzmann_init()
        for each outer step:
            image  = random_choice(dataset)
            theta_fast = copy(theta)
            run K inner Adam steps on image → theta_fast
            theta += outer_lr * (theta_fast - theta)   # nudge toward solution
    """

    LOG = logging.getLogger(__name__)

    def main(self):
        p = argparse.ArgumentParser(description="Reptile meta-trainer for SIREN init")
        p.add_argument("images_dir",              type=str,   nargs="?", help="directory tree of training images")
        p.add_argument("--image_list",            type=str,   help="text file with one image path per line (alternative to images_dir)")
        p.add_argument("-o", "--output",          type=str,   default="meta_init.npz")
        p.add_argument("--outer_steps",           type=int,   default=2002)
        p.add_argument("--inner_steps",           type=int,   default=17)
        p.add_argument("--inner_lr",              type=float, default=1e-3)
        p.add_argument("--outer_lr",              type=float, default=0.1)
        p.add_argument("--batch",                 type=int,   default=8118)
        p.add_argument("--size",                  type=int,   default=255,  help="resize images to NxN for meta-training")
        p.add_argument("--save_every",            type=int,   default=212)
        p.add_argument("--checkpoint",             type=str,   help="resume from checkpoint")
        # architecture — must match the runs you intend to warm-start
        p.add_argument("-s", "--model_size",      type=int,   default=16)
        p.add_argument("-c", "--model_count",     type=int,   default=4)
        p.add_argument("-q", "--coding",          type=int,   default=3)
        p.add_argument("-e", "--mapping",         choices=["polar", "fourier", "legacy"], default="polar")
        p.add_argument("-k", "--colorspace",      choices=["rgb", "ycbcr", "yuv"],        default="ycbcr")
        p.add_argument("-a", "--activation",      choices=["sine", "relu", "tanh"],        default="sine")
        p.add_argument("-f", "--four",            action="store_true")
        p.add_argument("-l", "--loss_fn",         choices=["mse", "huber", "l1"], default="mse")
        self.args = p.parse_args()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.LOG.info(f"device: {self.device}")

        if self.args.image_list:
            with open(self.args.image_list) as f:
                images = [l.strip() for l in f if l.strip() and os.path.isfile(l.strip())]
            self.LOG.info(f"dataset: {len(images)} images from {self.args.image_list}")
        elif self.args.images_dir:
            images = self._collect_images(self.args.images_dir)
            self.LOG.info(f"dataset: {len(images)} images in {self.args.images_dir}")
        else:
            raise RuntimeError("provide images_dir or --image_list")

        model = self._make_model()
        if self.args.checkpoint:
            self._load(model, self.args.checkpoint)
            self.LOG.info(f"resuming from {self.args.checkpoint}")

        self._reptile(model, images)

    # ------------------------------------------------------------------ #

    def _reptile(self, meta, images):
        args = self.args
        t0 = time.time()
        loss_ema = None

        for step in range(1, args.outer_steps + 1):
            path = random.choice(images)
            try:
                coords, colors = self._load_image(path)
            except Exception as e:
                self.LOG.warning(f"skipping {os.path.basename(path)}: {e}")
                continue

            fast = copy.deepcopy(meta)
            opt  = torch.optim.Adam(fast.parameters(), lr=args.inner_lr)
            n    = coords.shape[0]
            cap  = min(args.batch, n)

            for _ in range(args.inner_steps):
                idx   = torch.randperm(n, device=self.device)[:cap]
                pred  = fast(coords[idx])
                if args.activation == "sine":
                    pred = (pred + 1.0) / 2.0
                if args.loss_fn == "huber":
                    loss = F.huber_loss(pred, colors[idx])
                elif args.loss_fn == "l1":
                    loss = F.l1_loss(pred, colors[idx])
                else:
                    loss = F.mse_loss(pred, colors[idx])
                opt.zero_grad()
                loss.backward()
                opt.step()

            with torch.no_grad():
                for mp, fp in zip(meta.parameters(), fast.parameters()):
                    mp.data += args.outer_lr * (fp.data - mp.data)

            lv = loss.item()
            loss_ema = lv if loss_ema is None else 0.95 * loss_ema + 0.05 * lv

            if step % 20 == 0:
                elapsed  = time.time() - t0
                eta      = elapsed / step * (args.outer_steps - step)
                self.LOG.info(
                    f"step {step:>5}/{args.outer_steps}  "
                    f"loss={lv:.5f}  ema={loss_ema:.5f}  "
                    f"eta={eta/60:.1f}min"
                )

            if step % args.save_every == 0:
                self._save(meta, args.output)
                self.LOG.info(f"checkpoint → {args.output}")

        self._save(meta, args.output)
        self.LOG.info(f"done in {(time.time()-t0)/60:.1f}min → {args.output}")

    # ------------------------------------------------------------------ #

    def _make_model(self):
        a = self.args
        return Neuralistica(
            input_size  = 2 * 2 + a.coding * 4,
            output_size = 4 if a.four else 3,
            hidden_size = a.model_size,
            hidden_count= a.model_count,
            activation  = a.activation,
        ).to(self.device)

    def _load_image(self, path):
        a = self.args
        img  = Image.open(path).convert("RGB").resize((a.size, a.size), Image.LANCZOS)
        data = np.array(img) / 255.0

        if a.colorspace == "ycbcr":
            data = np.clip(np.dot(data, [
                [ 0.299,     0.587,     0.114   ],
                [-0.168736, -0.331264,  0.5     ],
                [ 0.5,      -0.418688, -0.081312],
            ]), 0, 1)
        elif a.colorspace == "yuv":
            data = np.clip(np.dot(data, [
                [ 0.299,    0.587,    0.114   ],
                [-0.14713, -0.28886,  0.436   ],
                [ 0.615,   -0.51499, -0.10001 ],
            ]), 0, 1)

        x, y   = np.linspace(-1, 1, a.size), np.linspace(-1, 1, a.size)
        xx, yy = np.meshgrid(x, y)
        coords = np.stack([xx, yy], axis=-1).reshape(-1, 2)
        colors = data.reshape(-1, 3)

        if a.four:
            mean   = np.mean(colors, axis=1, keepdims=True)
            colors = np.concatenate([colors, mean], axis=1)

        if a.coding > 0:
            coords = self._encode(coords)

        return (
            torch.tensor(coords, dtype=torch.float32, device=self.device),
            torch.tensor(colors, dtype=torch.float32, device=self.device),
        )

    def _encode(self, coords):
        a    = self.args
        x, y = coords[..., 0], coords[..., 1]

        if a.mapping == "polar":
            r     = np.sqrt(x**2 + y**2) + 1e-8
            s2    = np.stack([y/r, x/r], axis=-1)
            start = 0
        elif a.mapping == "fourier":
            f0    = np.pi
            s2    = np.stack([np.sin(f0*x), np.cos(f0*x)], axis=-1)
            start = 1
        else:
            q     = 22.0
            ang   = np.abs(((x*q)%1)-.5)*2 * np.abs(((y*q)%1)-.5)*2
            lng   = (np.sqrt(x**2+y**2)*q) % 1
            s2    = np.stack([ang, lng], axis=-1)
            start = 0

        parts = [coords, s2]
        for i in range(start, start + a.coding):
            freq = (2.0**i) * np.pi
            parts += [np.sin(freq*coords), np.cos(freq*coords)]
        return np.concatenate(parts, axis=-1)

    def _save(self, model, path):
        a = self.args
        config = {
            "model_size":  a.model_size,
            "model_count": a.model_count,
            "coding":      a.coding,
            "mapping":     a.mapping,
            "colorspace":  a.colorspace,
            "activation":  a.activation,
            "four":        getattr(a, "four", False),
            "loss_fn":     a.loss_fn,
        }
        weights = {k: v.detach().cpu().numpy() for k, v in model.named_parameters()}
        np.savez(path, __config__=np.array(json.dumps(config)), **weights)

    def _load(self, model, path):
        w = np.load(path)
        for name, param in model.named_parameters():
            param.data = torch.from_numpy(w[name]).to(self.device)

    def _collect_images(self, root):
        exts  = ("jpg", "jpeg", "png", "webp", "bmp")
        found = []
        for ext in exts:
            found.extend(glob.glob(os.path.join(root, "**", f"*.{ext}"), recursive=True))
            found.extend(glob.glob(os.path.join(root, "**", f"*.{ext.upper()}"), recursive=True))
        return list(set(found))


if __name__ == "__main__":
    MetaTrainer().main()
