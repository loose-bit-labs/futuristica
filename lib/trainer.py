from datetime import datetime
from PIL import Image

import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim.lr_scheduler as lr_scheduler

# for perceptual_loser
import torch.nn.functional as F
from torchvision import models, transforms

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

from model import Neuralistica


##
 #
 # Welcome to Futuristica
 #
 ##
class Futuristica:
    LOG = logging.getLogger(__name__)

    def __init__(self):
        losers = "mse l1 huber crossEntropy bce klDiv perceptual hybrid".split(" ")

        self.parser = argparse.ArgumentParser(description="This is Futuristica")
        self.parser.add_argument("-i", "--image",           type=str)
        self.parser.add_argument("-r", "--size",            type=int, default=512)
        self.parser.add_argument("-o", "--weights",         type=str, default="weights.npz")
        self.parser.add_argument("-g", "--generated",       type=str, default="generated_image.png")
        self.parser.add_argument("-t", "--training",        type=int, default=20)
        self.parser.add_argument("-q", "--coding",          type=int, default=3)
        self.parser.add_argument("-w", "--checkpoint",        type=str)
        self.parser.add_argument("--no_load_noise",           action="store_true", help="load checkpoint weights exactly, without perturbation")
        self.parser.add_argument("-s", "--model_size",      type=int, default=16)
        self.parser.add_argument("-c", "--model_count",     type=int, default=4)
        self.parser.add_argument("-l", "--loss_fn",         choices=losers, default="l1")
        self.parser.add_argument("-a", "--activation",      choices=["sine", "relu", "tanh"], default="sine")
        self.parser.add_argument("-e", "--mapping",          choices=["polar", "fourier", "legacy"], default="polar")
        self.parser.add_argument("-k", "--colorspace",      choices=["rgb", "ycbcr", "yuv"], default="ycbcr")
        self.parser.add_argument("-f", "--four",            action="store_true")
        self.parser.add_argument("-n", "--no_gui",          action="store_true")
        self.parser.add_argument("--steps",                 type=int, default=0)
        self.parser.add_argument("-b", "--batch",           type=int, default=32768)
        self.parser.add_argument("--plateau",               type=int, default=0)
        self.parser.add_argument("--progressive",           action="store_true", help="coarse-to-fine: train at size/4, size/2, then full size")


    def main(self):
        self.args = self.parser.parse_args()
        Futuristica.LOG.info(f'settings: {json.dumps(vars(self.args))}')

        self.last_image = None
        self.loss_history = []
        self.loss_bests = []
        self.create_plot()

        # Check if CUDA (GPU) is available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        Futuristica.LOG.info(f"Using device: {self.device}")

        import time
        t0 = time.time()

        coords, colors = self.load_image(self.args.image)
        model = self.train(coords, colors)

        self.export_weights(model, self.args.weights)
        self.generate_image(model, self.args.generated)
        self.eval_psnr(model)

        elapsed = int(time.time() - t0)
        h, rem = divmod(elapsed, 3600)
        m, s   = divmod(rem, 60)
        human  = (f"{h}h " if h else "") + (f"{m}m " if m or h else "") + f"{s}s"
        Futuristica.LOG.info(f"Runtime: {human} ({elapsed}s)")


    def load_image(self, image_path):
        size = self.args.size

        img = Image.open(image_path).convert("RGB")
        img = img.resize((size, size))  # Resize to speed up training
        img_data = np.array(img) / 255.0  # Convert to NumPy array and normalize
        if self.args.four:
            mean_rgb = np.mean(img_data.reshape(-1, 3), axis=1, keepdims=True)  # (h*w, 1)
        else:
            mean_rgb = None

        if 0 != self.args.steps:
            self.LOG.info(f"there are {self.args.steps} steps")
            img_data = np.floor(img_data * self.args.steps) / self.args.steps

        if "ycbcr" == self.args.colorspace:
            img_data = self.rgb_to_ycbcr(img_data)
        elif "yuv" == self.args.colorspace:
            img_data = self.rgb_to_yuv(img_data)

        h, w = img_data.shape[:2]

        x = np.linspace(-1, 1, w)
        y = np.linspace(-1, 1, h)
        xx, yy = np.meshgrid(x, y)
        coords = np.stack([xx, yy], axis=-1).reshape(-1, 2)  # (h*w, 2)
        colors = img_data.reshape(-1, 3)  # (h*w, 3)

        if not mean_rgb is None:
            colors = np.concatenate([colors, mean_rgb], axis=1)  # (h*w, 4)

        if self.args.coding > 0:
            coords = self.positional_encoding(coords, L=self.args.coding, mapping=self.args.mapping)

        # Move data to GPU
        return (torch.tensor(coords, dtype=torch.float32, device=self.device),
                torch.tensor(colors, dtype=torch.float32, device=self.device))


    def load_image_at_size(self, image_path, size):
        """Load image at an explicit size, ignoring self.args.size."""
        orig = self.args.size
        self.args.size = size
        result = self.load_image(image_path)
        self.args.size = orig
        return result

    def train(self, coords, colors):
        if not self.args.progressive:
            return self.train_back(coords, colors)

        full_size = self.args.size
        total_epochs = 5000 * self.args.training
        # stages: (fraction_of_total, resolution)
        stages = [
            (0.20, max(16, full_size // 4)),
            (0.30, max(32, full_size // 2)),
            (0.50, full_size),
        ]
        Futuristica.LOG.info(f"Progressive training: {[s[1] for s in stages]} -> {total_epochs} total epochs")
        return self.train_progressive(stages, total_epochs)


    def create_model(self, first=True):
        input_size = 2 * 2 + self.args.coding * 4
        output_size = 3
        if self.args.four:
            output_size = 4
        model = Neuralistica(
            input_size=input_size,
            output_size=output_size,
            hidden_size=self.args.model_size,
            hidden_count=self.args.model_count,
            activation=self.args.activation,
        ).to(self.device)
        if self.args.activation == "sine" and first:
            Futuristica.LOG.info("SIREN weight init applied")
        return model


    def get_loss_function(self, key=None):
        loss_fns = {
            "mse":           torch.nn.MSELoss,
            "l1":            torch.nn.L1Loss,
            "huber":         torch.nn.HuberLoss,
            "crossEntropy":  torch.nn.CrossEntropyLoss,
            "bce":           torch.nn.BCELoss,
            "klDiv":         torch.nn.KLDivLoss,
            "perceptual":    self.perceptual_loser,
            "hybrid":        self.hybrid_loser,
        }
        if key not in loss_fns:
            Futuristica.LOG.warning(f"Unknown loss function '{key}', defaulting to mse")
            key = "mse"
        return loss_fns[key]


    def train_progressive(self, stages, total_epochs):
        """Train with coarse-to-fine resolution stages, single continuous optimizer."""
        model = self.create_model()
        if self.args.checkpoint and '""' != self.args.checkpoint:
            model = self.load_weights(model, self.args.checkpoint)

        loss_fn   = self.get_loss_function(self.args.loss_fn)()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs, eta_min=1e-6)

        best_model  = None
        best_loss   = float('inf')
        imaged_last = 0
        imaged_too_soon = 50000
        use_sine    = self.args.activation == "sine"
        plateau     = self.args.plateau
        no_improve  = 0

        epoch = 0
        for frac, size in stages:
            stage_epochs = int(total_epochs * frac)
            coords, colors = self.load_image_at_size(self.args.image, size)
            n_pixels = coords.shape[0]
            batch    = min(self.args.batch, n_pixels)
            best_loss   = float('inf')
            best_model  = None
            imaged_last = 0
            no_improve  = 0
            Futuristica.LOG.info(f"Progressive stage: size={size}, epochs={stage_epochs}")

            for _ in range(stage_epochs):
                idx = torch.randperm(n_pixels, device=self.device)[:batch]
                optimizer.zero_grad()
                predictions = model(coords[idx])
                if use_sine:
                    predictions = (predictions + 1.0) / 2.0
                loss = loss_fn(predictions, colors[idx])
                loss.backward()
                optimizer.step()
                scheduler.step()

                loss_v = loss.item()
                self.loss_history.append(loss_v)

                if loss_v < best_loss:
                    best_loss  = loss_v
                    no_improve = 0
                    best_model = self.create_model(first=False)
                    best_model.load_state_dict(model.state_dict())
                    best_model.cpu()
                    self.loss_bests.append(best_loss)
                    if imaged_last == 0 or epoch - imaged_last > imaged_too_soon:
                        self.generate_image(model, self.args.generated, True)
                        self.export_weights(model, self.args.weights)
                        imaged_last = epoch
                        self.update_plot(epoch)
                else:
                    no_improve += 1
                    if plateau and no_improve >= plateau:
                        Futuristica.LOG.info(f"Epoch {epoch}: plateau, stopping")
                        best_model.to(self.device)
                        return best_model

                if epoch % 1000 == 0:
                    lr = scheduler.get_last_lr()[0]
                    self.update_plot(epoch)
                    Futuristica.LOG.info(f"Epoch {epoch}[{int(epoch*100/total_epochs)}%]: Loss: {loss_v:.6f} vs ({best_loss:.6f}), LR:{lr:.6e}")

                epoch += 1

        Futuristica.LOG.info(f"Training complete: {best_loss:.6f}")
        best_model.to(self.device)
        return best_model

    def train_back(self, coords, colors):
        model = self.create_model()

        if self.args.checkpoint and '""' != self.args.checkpoint:
            model = self.load_weights(model, self.args.checkpoint)

        best_model      = None
        best_loss       = float('inf')
        imaged_last     = 0
        imaged_too_soon = 50000  # minimum epochs between checkpoint images

        loss_fn   = self.get_loss_function(self.args.loss_fn)()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        epochs    = 5000 * self.args.training

        # Cosine annealing over the full run — starts warm, cools to near-zero.
        # Previously T_max was 100 — cycled thousands of times, effectively useless.
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

        Futuristica.LOG.info(f"Training for 5k x {self.args.training} (minutes) -> {epochs}")

        # SIREN outputs [-1,1]; colour targets are [0,1] — remap before loss.
        use_sine  = self.args.activation == "sine"
        n_pixels  = coords.shape[0]
        batch     = min(self.args.batch, n_pixels)  # clamp to dataset size
        plateau   = self.args.plateau
        no_improve = 0

        for epoch in range(epochs):
            # random minibatch each step
            idx = torch.randperm(n_pixels, device=self.device)[:batch]
            batch_coords = coords[idx]
            batch_colors = colors[idx]

            optimizer.zero_grad()
            predictions = model(batch_coords)
            if use_sine:
                predictions = (predictions + 1.0) / 2.0
            loss = loss_fn(predictions, batch_colors)
            loss.backward()
            optimizer.step()
            scheduler.step()

            loss_v = loss.item()
            self.loss_history.append(loss_v)
            now = f"Epoch {epoch}[{int(epoch * 100 / epochs)}%]"

            if loss_v < best_loss:
                best_loss  = loss_v
                no_improve = 0
                best_model = self.create_model(first=False)
                best_model.load_state_dict(model.state_dict())
                best_model.cpu()
                self.loss_bests.append(best_loss)
                if imaged_last == 0 or epoch - imaged_last > imaged_too_soon:
                    self.generate_image(model, self.args.generated, True)
                    self.export_weights(model, self.args.weights)
                    imaged_last = epoch
                    self.update_plot(epoch)
            else:
                no_improve += 1
                if plateau and no_improve >= plateau:
                    Futuristica.LOG.info(f"{now}: plateau after {plateau} epochs, stopping")
                    break

            if epoch % 1000 == 0:
                lr = scheduler.get_last_lr()[0]
                self.update_plot(epoch)
                Futuristica.LOG.info(f"{now}: Loss: {loss_v:.6f} vs ({best_loss:.6f}), LR:{lr:.6e}")

        Futuristica.LOG.info(f"Training complete: {best_loss:.6f}")
        best_model.to(self.device)
        return best_model


    def export_weights(self, model, filename="weights.npz"):
        Futuristica.LOG.info(f"Saving weights to {filename}")
        weights = {name: param.detach().cpu().numpy() for name, param in model.named_parameters()}
        config = {
            "model_size":  self.args.model_size,
            "model_count": self.args.model_count,
            "coding":      self.args.coding,
            "mapping":     self.args.mapping,
            "colorspace":  self.args.colorspace,
            "activation":  self.args.activation,
            "four":        self.args.four,
            "loss_fn":     self.args.loss_fn,
            "size":        self.args.size,
        }
        np.savez(filename, __config__=np.array(json.dumps(config)), **weights)
        Futuristica.LOG.info(f"Saved weights to {filename}")


    def load_weights(self, model, filename="weights.npz"):
        if not filename:
            Futuristica.LOG.warning(f"no weights to load")
            return model
        Futuristica.LOG.info(f"Loading weights from {filename}")
        weights = np.load(filename)
        if '__config__' in weights:
            config = json.loads(str(weights['__config__']))
            Futuristica.check_compat(config, self.args, filename)
        noise_scale = 0.0 if self.args.no_load_noise else 0.01
        for name, param in model.named_parameters():
            ww = weights[name].copy()
            if noise_scale:
                ww += np.random.uniform(low=-noise_scale, high=noise_scale, size=ww.shape)
            param.data = torch.from_numpy(ww).to(param.device)
        Futuristica.LOG.info(f"Loaded weights from {filename}")
        return model

    @staticmethod
    def check_compat(config, args, filename=""):
        arch_keys = ["model_size", "model_count", "coding", "four"]
        mismatches = []
        for k in arch_keys:
            stored = config.get(k)
            current = getattr(args, k, None)
            if stored is not None and current is not None and stored != current:
                mismatches.append(f"  {k}: checkpoint has {stored!r}, got {current!r}")
        if mismatches:
            where = f" ({filename})" if filename else ""
            raise ValueError(f"Checkpoint architecture mismatch{where}:\n" + "\n".join(mismatches))

    @staticmethod
    def read_config(filename):
        """Return the __config__ dict from an npz, or None if not present."""
        weights = np.load(filename)
        if '__config__' in weights:
            return json.loads(str(weights['__config__']))
        return None


    def generate_image(self, model, filename = "generated_image.png", timestamp = False):
        size = self.args.size

        x_coords = np.linspace(-1, 1, size)
        y_coords = np.linspace(-1, 1, size)
        x_grid, y_grid = np.meshgrid(x_coords, y_coords)

        # Flatten the input coordinates
        inputs = np.stack((x_grid.flatten(), y_grid.flatten()), axis=1)

        if self.args.coding > 0:
            inputs = self.positional_encoding(inputs, L=self.args.coding, mapping=self.args.mapping)

        # Convert inputs to PyTorch tensor
        inputs_tensor = torch.from_numpy(inputs).float()

        # Move the model to the GPU (if available)
        model.to(self.device)

        # Move the input tensor to the GPU
        inputs_tensor = inputs_tensor.to(self.device)

        # Pass inputs through the model
        outputs = model(inputs_tensor)

        # Reshape and normalize the output.
        # relu/tanh networks use Sigmoid so output is already [0,1].
        # sine (SIREN) networks output in [-1,1] — remap to [0,1] first.
        raw = outputs.reshape(size, size, -1).detach().cpu().numpy()
        if self.args.activation == "sine":
            raw = (raw + 1.0) / 2.0  # [-1,1] -> [0,1]
        raw = np.clip(raw, 0.0, 1.0)
        outputs = raw * 255

        if "ycbcr" == self.args.colorspace:
            outputs = self.ycbcr_to_rgb(outputs);
        elif "yuv" == self.args.colorspace:
            outputs = self.yuv_to_rgb(outputs);

        if self.args.four:
            outputs = self.ten_four(outputs, False)

        # Convert output to image
        self.last_image = outputs
        image = Image.fromarray(outputs.astype(np.uint8))

        if timestamp:
            t = datetime.now()
            now = t.strftime('%Y-%m-%d+%H-%M-%S')
            extension = filename.rsplit('.', 1)[1]
            name = filename.rsplit('.', 1)[0]
            filename = f"{name}-{now}.{extension}"

        # Save the image
        image.save(filename)
        Futuristica.LOG.info(f"Generated image as {filename}")


    def eval_psnr(self, model):
        """Reconstruct the full image and compute PSNR against the original in RGB [0,1].

        Logged as 'Eval PSNR: XX.XXdB' so grid-html-maker can grep it from train.log.
        This is a fixed metric independent of the training loss function, making
        results comparable across l1/mse/huber/etc. runs.
        """
        size = self.args.size

        # original image in RGB [0,1]
        orig = np.array(
            Image.open(self.args.image).convert("RGB").resize((size, size))
        ) / 255.0  # (H, W, 3)

        # reconstruct via the model
        x = np.linspace(-1, 1, size)
        y = np.linspace(-1, 1, size)
        xx, yy = np.meshgrid(x, y)
        inputs = np.stack((xx.flatten(), yy.flatten()), axis=1)
        if self.args.coding > 0:
            inputs = self.positional_encoding(inputs, L=self.args.coding, mapping=self.args.mapping)

        with torch.no_grad():
            t = torch.from_numpy(inputs).float().to(self.device)
            out = model(t)
            if self.args.activation == "sine":
                out = (out + 1.0) / 2.0
            raw = np.clip(out.cpu().numpy().reshape(size, size, -1), 0.0, 1.0)

        # convert back to RGB [0,1], preserving extra channels (e.g. alpha from --four)
        alpha = raw[..., 3:] if raw.shape[-1] > 3 else None
        if self.args.colorspace == "ycbcr":
            raw = np.clip(np.dot(raw[..., :3],
                [[1, 0, 1.402], [1, -0.344136, -0.714136], [1, 1.772, 0]]), 0, 1)
        elif self.args.colorspace == "yuv":
            raw = np.clip(np.dot(raw[..., :3],
                [[1, 0, 1.13983], [1, -0.39465, -0.58060], [1, 2.03211, 0]]), 0, 1)
        if alpha is not None:
            raw = np.concatenate([raw, alpha], axis=-1)

        if self.args.four:
            rgb = raw[..., :3]
            gray = raw[..., 3:4]
            mean = np.where(np.mean(rgb, axis=-1, keepdims=True) == 0, 1e-6,
                            np.mean(rgb, axis=-1, keepdims=True))
            raw = np.clip(rgb * (gray / mean), 0, 1)

        mse = np.mean((orig - raw[..., :3]) ** 2)
        psnr = 20 * np.log10(1.0 / np.sqrt(mse)) if mse > 0 else float('inf')
        Futuristica.LOG.info(f"Eval PSNR: {psnr:.2f}dB")
        return psnr


    def ten_four(self, outputs, needs_reshaping = True):
        size = self.args.size
        if needs_reshaping:
            outputs = outputs.reshape(size, size, 4).detach().cpu().numpy()

        outputs = outputs / 255

        # Split "RGB" and grayscale
        rgb = outputs[..., :3]
        grayscale = outputs[..., 3:4]  # keep dims for broadcasting

        # Calculate mean of RGB, avoiding division by zero
        mean = np.mean(rgb, axis=-1, keepdims=True)
        mean = np.where(mean == 0, 1e-6, mean)  # Prevent division by zero

        ratio = grayscale / mean

        return rgb * ratio * 255


    def positional_encoding(self, coords, L=3, mapping="polar"):
        """Encode (x,y) coordinates into a fixed-width feature vector.

        All three mappings produce the same total width (2 + 2 + L*4)
        so the network input size never changes when you switch --mapping.
        The first 2 slots are always raw [x, y].
        Slots 2-3 are the "spatial hint" — what differs per mapping.
        Slots 4+ are always Fourier sin/cos bands.

        polar   (default)
            slots 2-3: [sin(θ), cos(θ)]  where θ = atan2(y,x)
            Encodes direction without any discontinuity — sin/cos of angle
            wraps smoothly everywhere unlike raw atan2 which jumps at the
            negative x-axis.  radius intentionally dropped; the Fourier
            bands already encode scale implicitly.

        fourier
            slots 2-3: first Fourier band (sin/cos of freq-0 * x, y)
            Pure spectral encoding, no spatial structure hint.
            Identical to the classic NeRF positional encoding.
            Use L+1 effective bands; the loop starts at band 1.

        legacy
            slots 2-3: abs(fract()) kink-based terms from the original code
            Kept for reproducibility. The kinks cause gradient noise and
            banding artifacts — not recommended for new runs.

        Output width = 2 + 2 + L*4 = 16 for L=3 (matches default input_size).
        translate.py must mirror this exactly — both controlled by --mapping.
        """
        x, y = coords[..., 0], coords[..., 1]

        if mapping == "polar":
            # sin/cos of angle — globally continuous, no atan2 jump
            r     = np.sqrt(x**2 + y**2) + 1e-8   # avoid div-by-zero
            s2    = np.stack([y / r, x / r], axis=-1)  # [sin θ, cos θ]
            bands_start = 0  # use all L bands

        elif mapping == "fourier":
            # first Fourier band fills the hint slots; loop starts at band 1
            freq0 = np.pi  # 2^0 * pi
            s2    = np.stack([np.sin(freq0 * x), np.cos(freq0 * x)], axis=-1)
            bands_start = 1  # skip band 0, already used above

        else:  # legacy
            q   = 22.0
            x2  = np.abs(((x * q) % 1) - 0.5) * 2
            y2  = np.abs(((y * q) % 1) - 0.5) * 2
            ang = x2 * y2
            lng = (np.sqrt(x**2 + y**2) * q) % 1
            s2  = np.stack([ang, lng], axis=-1)
            bands_start = 0

        encodings = [coords, s2]

        for i in range(bands_start, bands_start + L):
            freq = (2.0 ** i) * np.pi
            encodings.append(np.sin(freq * coords))
            encodings.append(np.cos(freq * coords))

        return np.concatenate(encodings, axis=-1)


    # never had much luck with this, very slow
    def hybrid_loser(self, weight = 100, penalty=.001):
        perceptual_loss = self.perceptual_loser()
        def hybrid_loss(pred, target):
            perceptual = perceptual_loss(pred, target)
            pixelwise = F.mse_loss(pred, target)
            return perceptual * penalty + weight * pixelwise
        return hybrid_loss


    # never had much luck with this, very slow
    def perceptual_loser(self):
        size = self.args.size
        # Load VGG16 properly with weights
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features[:16].eval().to(self.device)

        # VGG expects normalized input
        vgg_normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        def perceptual_loss(pred, target):
            """Compute perceptual loss between predicted and target images"""

            # Reshape flat tensor back into image format (Batch, Channels, Height, Width)
            pred = pred.view(1, 3, size, size)  # (batch=1, channels=3, height=size, width=size)
            target = target.view(1, 3, size, size)

            # Normalize before passing into VGG
            pred = vgg_normalize(pred)
            target = vgg_normalize(target)

            # Extract deep features
            pred_features = vgg(pred)
            target_features = vgg(target)

            # Compute perceptual loss
            return F.mse_loss(pred_features, target_features)
        return perceptual_loss


    def rgb_to_ycbcr(self, img_data):
        return np.clip(
            np.dot(img_data[:, :, :3], [
                [ 0.299,     0.587,     0.114],
                [-0.168736, -0.331264,  0.5],
                [ 0.5,      -0.418688, -0.081312]
            ]), 0, 1
        )

    def ycbcr_to_rgb(self, img_data):
        transformation_matrix = np.array([[1, 0, 1.402], [1, -0.344136, -0.714136], [1, 1.772, 0]])
        rgb_data = np.clip(np.dot(img_data[..., :3], transformation_matrix), 0, 255)
        if img_data.shape[-1] == 4:
            rgb_data = np.concatenate((rgb_data, img_data[..., 3:]), axis=-1)
        return rgb_data


    def rgb_to_yuv(self, img_data):
        return np.clip(
            np.dot(img_data[:, :, :3], [
                [0.299, 0.587, 0.114],
                [-0.14713, -0.28886, 0.436],
                [0.615, -0.51499, -0.10001]
            ]),
            0, 1
        )


    def yuv_to_rgb(self, img_data):
        return np.clip(
            np.dot(img_data, [
                [1, 0, 1.13983],
                [1, -0.39465, -0.58060],
                [1, 2.03211, 0]
            ]),
            0, 255
        )


    def create_plot(self):
        if self.args.no_gui:
            self.LOG.info("no gui...")
            return
        plt.ion()

        self.fig, (self.ax_loss, self.ax_image) = plt.subplots(1, 2, figsize=(12, 5), num="futuristica")

        # Loss plot
        self.ax_loss.set_title("Training Loss Over Time")
        self.ax_loss.set_xlabel("Epoch")
        self.ax_loss.set_ylabel("Loss")

        # Image display
        self.ax_image.set_title("Latest Generated Image")
        size = self.args.size
        self.image_plot = self.ax_image.imshow(np.zeros((size, size, 3))) # Placeholder image


    def update_plot(self, epoch = -3e3):
        if self.args.no_gui:
            return
        last = self.loss_history[-1]
        # Update loss plot
        self.ax_loss.clear()
        self.ax_loss.set_title(f"Training Loss Over Time: {last}")
        self.ax_loss.set_xlabel("Epoch")
        self.ax_loss.set_ylabel("Loss")
        moving_average = np.convolve(self.loss_history, np.ones(10)/10, mode='valid')
        self.ax_loss.plot(moving_average, color='blue', linestyle='--', label='Training Loss')
        self.ax_loss.plot(self.loss_bests, color='green', label="Best Loss")
        self.ax_loss.legend()

        # Update latest image
        if not self.last_image is None:
            self.image_plot.set_data(self.last_image / 255)
            self.ax_image.set_title(f"Latest Generated Image (Epoch {epoch})")

        # Redraw canvas
        plt.draw()
        plt.pause(0.1)  # Small pause to allow update
