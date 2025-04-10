#!/usr/bin/env python
#############################################################################

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

# Define a simple neural network model
# It seems like GLSL get's maxed out anything 32x32 or greater
# Trying to go deep (giggetty)
class Neuralistica(torch.nn.Module):
    def __init__(self, input_size=16, output_size=3, hidden_size=16, hidden_count=4):
        super().__init__()
        layers = []

        layers.append(torch.nn.Linear(input_size, hidden_size))
        layers.append(torch.nn.ReLU())
        for _ in range(hidden_count):
            layers.append(torch.nn.Linear(hidden_size, hidden_size))
            layers.append(torch.nn.ReLU())
        layers.append(torch.nn.Linear(hidden_size, output_size))
        layers.append(torch.nn.Sigmoid())
        self.layers = torch.nn.Sequential(*layers)


    def forward(self, x):
        return self.layers(x)
# end of class Neuralistica


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
        self.parser.add_argument("--image",       type=str)
        self.parser.add_argument("--size",        type=int, default=512)
        self.parser.add_argument("--weights",     type=str, default="weights.npz")
        self.parser.add_argument("--generated",   type=str, default="generated_image.png")
        self.parser.add_argument("--training",    type=int, default=20)
        self.parser.add_argument("--coding",      type=int, default=2)
        self.parser.add_argument("--ckp",         type=str)
        self.parser.add_argument("--model_size",  type=int, default=16)
        self.parser.add_argument("--model_count", type=int, default=4)
        self.parser.add_argument("--loss_fn",     choices=losers, default="mse")
        self.parser.add_argument("--colorspace",  choices=["rgb", "ycbcr", "yuv"], default="ycbcr")
        self.parser.add_argument("--rollback_too_soon",     type=int, default=100)
        self.parser.add_argument("--rollback_way_too_long", type=int, default=500)
    # end of __init__


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

        coords, colors = self.load_image(self.args.image)
        model = self.train(coords, colors)

        self.export_weights(model, self.args.weights)
        self.generate_image(model, self.args.generated)
    # end of main


    def load_image(self, image_path):
        size = self.args.size

        img = Image.open(image_path).convert("RGB")
        img = img.resize((size, size))  # Resize to speed up training
        img_data = np.array(img) / 255.0  # Convert to NumPy array and normalize
        # this is terrible: img_data = np.power(img_data, 2.2)  # Apply gamma correction

        #def load_image(self, image_path):
        #    ...
        #    img_data = np.array(img) / 255.0  # Convert to NumPy array and normalize
        #    ...
        #    colors = img_data.reshape(-1, 3)  # (h*w, 3)
        #
        #    # Calculate mean of RGB values
        #    mean_rgb = np.mean(colors, axis=1, keepdims=True)  # (h*w, 1)
        #
        #    # Append mean RGB value to colors
        #    colors_with_mean = np.concatenate([colors, mean_rgb], axis=1)  # (h*w, 4)
        #
        #    ...
        #    return (torch.tensor(coords, dtype=torch.float32, device=self.device),
        #            torch.tensor(colors_with_mean, dtype=torch.float32, device=self.device))

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

        if self.args.coding > 0:
            coords = self.positional_encoding(coords, L=self.args.coding) #10)

        # Move data to GPU
        return (torch.tensor(coords, dtype=torch.float32, device=self.device), 
                torch.tensor(colors, dtype=torch.float32, device=self.device))
    # end of load_image


    # TODO: make a training flag
    def train(self, coords, colors):
        return self.train_back(coords, colors)
        return self.best_of_train(coords, colors)
        return self.og_train(coords, colors)


    def create_model(self):
        input_size = 2 * 2 + self.args.coding * 4 
        return Neuralistica(
            input_size=input_size, 
            output_size=3, 
            hidden_size=self.args.model_size, 
            hidden_count=self.args.model_count,
        ).to(self.device)


    def get_loss_function(self, key=None):
        loss_fns = {
            "mse":           torch.nn.MSELoss,
            "l1":            torch.nn.L1Loss,
            "huber":         torch.nn.HuberLoss,
            "crossEntropy":  torch.nn.CrossEntropyLoss,
            "bce":           torch.nn.BCELoss,
            "klDiv":         torch.nn.KLDivLoss,
            #"marginRanking": torch.nn.MarginRankingLoss,
            #"ctc":           torch.nn.CTCLoss,
            "perceptual":    self.perceptual_loser,
            "hybrid":        self.hybrid_loser,
        }
        if None:
            return list(loss_fns.keys())
        else:
            if not key in loss_fns:
                Futuristica.LOG.warning(f"Unknown loss function '{key}', defaulting to 'pixel'");
                key = "mse"
            return loss_fns[key]


    # this version supports rollback but can take longer
    def train_back(self, coords, colors):
        # Initialize model and move it to GPU
        model = self.create_model()

        # If specified load the old checkpoint
    
        if self.args.ckp:
            model = self.load_weights(model, self.args.ckp)

        # Track the lowest loss
        
        best_model = None
        best_loss = float('inf')
        spammed = 0
        imaged_too_soon = 100 * 10 # don't make too many images...
        loss_fn = self.get_loss_function(self.args.loss_fn)()

        ################################################################################
        # set up the optimizer and scheduler

        #optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)
        ################################################################################


        # Training loop 
        epochs = 5000 * self.args.training
        Futuristica.LOG.info(f"Training for 5k x {self.args.training} (minutes) -> {epochs}");

        # don't rollback too often

        # how long to go without improvement since the last checkpoint 
        rollback_too_soon = self.args.rollback_too_soon
        # but.. that only kicks if there was a checkpoint not too long ago
        # if we go a *really* long time without checkpoint and without rollback
        # then roll back cuz we wandered way off target
        rollback_way_too_long = rollback_too_soon * self.args.rollback_way_too_long

        # counters and tracking

        rollback_last = 0
        has_checkpoint = False
        imaged_last = 0

        # main training loop

        for epoch in range(epochs):
            optimizer.zero_grad()
            predictions = model(coords)  # Forward pass
            loss = loss_fn(predictions, colors)  # Compute loss
            loss.backward()  # Backpropagation
            optimizer.step()  # Update weights
            scheduler.step()  # Update learning rate

            loss_v = loss.item()
            self.loss_history.append(loss_v)
            percent = (loss_v - best_loss) / best_loss * 100

            now = f"Epoch {epoch}[{int(epoch * 100 / epochs)}%]"

            if loss_v < best_loss:
                # When the loss improves we save the model
                spammed = spammed + 1
                if spammed < 4:
                    Futuristica.LOG.info(f"{now} Checkpoint since is {loss_v:.6f} vs {best_loss:.6f}, {percent:.4f}%")
                best_loss = loss_v
                best_model = self.create_model() 
                best_model.load_state_dict(model.state_dict())  # Copy the model's state dict
                best_model.cpu()  # Move the model to CPU
                has_checkpoint = True
                self.loss_bests.append(best_loss)
                if 0 == imaged_last or epoch - imaged_last > imaged_too_soon:
                    self.generate_image(model, self.args.generated, True)
                    self.export_weights(model, self.args.weights) # warum nicht?
                    imaged_last = epoch
                    self.update_plot(epoch)
            elif loss_v > best_loss:
                # If the loss is getting worse for a while, we roll back
                rollback_since_last = epoch - rollback_last
                # if it's been a really long time since the last checkpoint and rollback
                # training has gone way off course
                if rollback_since_last > rollback_way_too_long and not has_checkpoint:
                    Futuristica.LOG.info(f"{now}: PUNT: {rollback_since_last} way is too long!")
                    has_checkpoint = True
                if rollback_since_last > rollback_too_soon and has_checkpoint:
                    Futuristica.LOG.warning(f"{now}: Rollback from {loss_v:.6f} to {best_loss:.6f}, {percent:.4f}%")
                    model.load_state_dict(best_model.state_dict())  # Load the previous best state dict
                    #model.cuda()  # Move the model back to GPU
                    model.to(self.device) # Move the model back to GPU
                    rollback_last = epoch
                    has_checkpoint = False # at least not recently...
                    spammed = 0
            else:
                spammed = 0
            
            if epoch % 1000 == 0:
                lr = scheduler.get_last_lr()[0]
                self.update_plot(epoch)
                Futuristica.LOG.info(f"{now}: Loss: {loss_v:.6f} vs ({best_loss:.6f}), LR:{lr:.6e}")
                spammed = 0

        Futuristica.LOG.info(f"Training complete: {best_loss:.6f}")

        best_model.to(self.device)
        return best_model
    #end of train


    # this version is supposed to just keep the lowest lost model it sees
    # FIXME: untested! 
    def best_of_train(self, coords, colors):
        # Initialize model and move it to GPU
        model = self.create_model()

        lowest = 1
        best = model.state_dict().copy()

        # Define loss function and optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        loss_fn = torch.nn.MSELoss()

        # Training loop 
        epochs = 5000 * self.args.training
        Futuristica.LOG.info(f"Training for 5k x {self.args.training} (minutes) -> {epochs}");

        for epoch in range(epochs):
            optimizer.zero_grad()
            predictions = model(coords)  # Forward pass
            loss = loss_fn(predictions, colors)  # Compute loss
            loss.backward()  # Backpropagation
            optimizer.step()  # Update weights
            
            if epoch % 1000 == 0:
                Futuristica.LOG.info(f"Epoch {epoch}: Loss = {loss.item():.6f}")
                if loss.item() < lowest:
                    Futuristica.LOG.info(f"New lowest is {loss.item():.6f} vs {lowest:.6f}")
                    lowest = loss.item()
                    best = model.state_dict().copy()

        Futuristica.LOG.info("Training complete!")

        model.load_state_dict(best)  # Load the best state dict
        model.to(self.device)
        return model
    #end of train


    # the og! this just does what it does!
    def og_train(self, coords, colors):
        # Initialize model and move it to GPU
        model = self.create_model()

        # Define loss function and optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        loss_fn = torch.nn.MSELoss()

        # Training loop 
        epochs = 5000 * self.args.training
        Futuristica.LOG.info(f"Training for 5k x {self.args.training} (minutes) -> {epochs}");

        for epoch in range(epochs):
            optimizer.zero_grad()
            predictions = model(coords)  # Forward pass
            loss = loss_fn(predictions, colors)  # Compute loss
            loss.backward()  # Backpropagation
            optimizer.step()  # Update weights
            
            if epoch % 1000 == 0:
                Futuristica.LOG.info(f"Epoch {epoch}: Loss = {loss.item():.6f}")

        Futuristica.LOG.info("Training complete!")
        return model
    #end of train


    # save out the results, you can run translate.py 
    # on the exported model 
    def export_weights(self, model, filename="weights.npz"):
        Futuristica.LOG.info(f"Saving weights to {filename}")
        weights = {}
        for i, (name, param) in enumerate(model.named_parameters()):
            weights[name] = param.detach().cpu().numpy()  # Convert to NumPy
        np.savez(filename, **weights)
        Futuristica.LOG.info(f"Saved weights to {filename}")
    #end of export_weights


    def load_weights(self, model, filename="weights.npz"):
        Futuristica.LOG.info(f"Loading weights from {filename}")
        weights = np.load(filename)
        noise_scale = .001
        noise_scale = .077
        noise_scale = .01
        for name, param in model.named_parameters():
            ww = weights[name]
            #ww += torch.randn_like(ww) * noise_scale
            ww += np.random.uniform(low=-noise_scale, high=noise_scale, size=ww.shape)


            param.data = torch.from_numpy(ww).to(param.device)
            #param.data = torch.from_numpy(weights[name]).to(param.device)
            #weights += torch.randn_like(weights) * noise_scale
        Futuristica.LOG.info(f"Loaded weights from {filename}")
        return model
    #end of load_weights


    def generate_image(self, model, filename = "generated_image.png", timestamp = False):
        size = self.args.size

        x_coords = np.linspace(-1, 1, size)
        y_coords = np.linspace(-1, 1, size)
        x_grid, y_grid = np.meshgrid(x_coords, y_coords)

        # Flatten the input coordinates
        inputs = np.stack((x_grid.flatten(), y_grid.flatten()), axis=1)

        if self.args.coding > 0:
            inputs = self.positional_encoding(inputs, L=self.args.coding)

        # Convert inputs to PyTorch tensor
        inputs_tensor = torch.from_numpy(inputs).float()

        # Move the model to the GPU (if available)
        model.to(self.device)

        # Move the input tensor to the GPU
        inputs_tensor = inputs_tensor.to(self.device)

        # Pass inputs through the model
        outputs = model(inputs_tensor)

        # Reshape and normalize the output
        outputs = outputs.reshape(size, size, 3).detach().cpu().numpy() * 255

        if "ycbcr" == self.args.colorspace:
            outputs = self.ycbcr_to_rgb(outputs);
        elif "yuv" == self.args.colorspace:
            outputs = self.yuv_to_rgb(outputs);

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
    # end of generated_image


    """Encodes (x, y) coordinates into a high-dimensional space"""
    # this seems to help learn sharp edges, but creates artifacts too
    def positional_encoding(self, coords, L=3):
        encodings = [coords]

        x, y = coords[..., 0], coords[..., 1]
        angle = x / (np.abs(y)+.001)
        #np.arctan2(y, x) * (1.01-x*y) * 33 # the discontinuity caused noticable artifacts
        
        q = 22
        x2 = np.abs(((x * q)%1)-.5) *2
        y2 = np.abs(((y * q)%1)-.5) *2
        angle = x2 * y2
        length = (np.sqrt(x**2 + y**2) * q )%1

        encodings = [coords, np.stack([angle, length], axis=-1)]

        for i in range(L):
            if i % 2:
                # this helps curvy shapes
                encodings.append(np.sin((2.0 ** i) * np.pi * coords))
                encodings.append(np.cos((2.0 ** i) * np.pi * coords))
            else:
                # this helps learn straight edges
                encodings.append(np.abs((2.0 ** (i+.0) - .5) * coords))
                encodings.append(np.abs((2.0 ** (i+.5) - .5) * coords))
        return np.concatenate(encodings, axis=-1)


    # this helped a lot
    def old_encoding(self, coords, L=10):
        """Encodes (x, y) coordinates into a high-dimensional space"""
        encodings = [coords]
        for i in range(L):
            encodings.append(np.sin((2.0 ** i) * np.pi * coords))
            encodings.append(np.cos((2.0 ** i) * np.pi * coords))
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


    def downsample_ycbcr(self, ycbcr_data, factor=2):
        # Separate the Y, Cb, and Cr channels
        y_channel = ycbcr_data[:, :, 0]
        cb_channel = ycbcr_data[:, :, 1]
        cr_channel = ycbcr_data[:, :, 2]

        # Downsample the CbCr channels by a factor of 2
        cb_downsampled = Image.fromarray(cb_channel.astype(np.uint8))
        cb_downsampled = cb_downsampled.resize((cb_downsampled.width // factor, cb_downsampled.height // factor), Image.BICUBIC)
        cb_downsampled = np.array(cb_downsampled)

        cr_downsampled = Image.fromarray(cr_channel.astype(np.uint8))
        cr_downsampled = cr_downsampled.resize((cr_downsampled.width // factor, cr_downsampled.height // factor), Image.BICUBIC)
        cr_downsampled = np.array(cr_downsampled)

        # Combine the Y, Cb, and Cr channels back together
        ycbcr_downsampled = np.stack((y_channel, cb_downsampled, cr_downsampled), axis=2)
        return ycbcr_downsampled


    def upsample_ycbcr(self, ycbcr_upsampled, factor=2):
        # Separate the Y, Cb, and Cr channels
        y_channel  = ycbcr_upsampled[:, :, 0]
        cb_channel = ycbcr_upsampled[:, :, 1]
        cr_channel = ycbcr_upsampled[:, :, 2]

        # Upsample the CbCr channels by a factor of 2
        cb_upsampled = Image.fromarray(cb_channel.astype(np.uint8))
        cb_upsampled = cb_upsampled.resize((cb_upsampled.width * factor, cb_upsampled.height * factor), Image.BICUBIC)
        cb_upsampled = np.array(cb_upsampled)

        cr_upsampled = Image.fromarray(cr_channel.astype(np.uint8))
        cr_upsampled = cr_upsampled.resize((cr_upsampled.width * factor, cr_upsampled.height * factor), Image.BICUBIC)
        cr_upsampled = np.array(cr_upsampled)

        # Combine the Y, Cb, and Cr channels back together
        ycbcr_upsampled = np.stack((y_channel, cb_upsampled, cr_upsampled), axis=2)
        return ycbcr_upsampled


    def rgb_to_ycbcr(self, img_data):
        return np.clip(
            np.dot(img_data[:, :, :3], [
                [ 0.299,     0.587,     0.114],
                [-0.168736, -0.331264,  0.5],
                [ 0.5,      -0.418688, -0.081312]
            ]), 0, 1
        )


    def ycbcr_to_rgb(sef, img_data):
        return np.clip(
            np.dot(img_data, [
                [1, 0, 1.402], 
                [1, -0.344136, -0.714136], 
                [1, 1.772, 0]
            ]), 0, 255
        )
        # return mat3(1, 0, 1.402, 1, -0.344136, -0.714136, 1, 1.772, 0.) * color;


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
        # return mat3(1, 0, 1.13983, 1, -0.39465, -0.58060, 1, 2.03211, 0) * color;


    def create_plot(self):
        plt.ion()
        
        self.fig, (self.ax_loss, self.ax_image) = plt.subplots(1, 2, figsize=(12, 5), num="futuristica")
        self.losses = []

        # Loss plot
        self.ax_loss.set_title("Training Loss Over Time")
        self.ax_loss.set_xlabel("Epoch")
        self.ax_loss.set_ylabel("Loss")
        
        # Image display
        self.ax_image.set_title("Latest Generated Image")
        size = self.args.size
        self.image_plot = self.ax_image.imshow(np.zeros((size, size, 3))) # Placeholder image
        

    def update_plot(self, epoch = -3e3):
        last = self.loss_history[-1]
        # Update loss plot
        self.ax_loss.clear()
        self.ax_loss.set_title(f"Training Loss Over Time: {last}")
        self.ax_loss.set_xlabel("Epoch")
        self.ax_loss.set_ylabel("Loss")
        if not True:
            self.ax_loss.plot(range(1, len(self.loss_history) + 1), self.loss_history, marker='o', linestyle='-')
        else:
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


# end of class Futuristica

if __name__ == "__main__":
    Futuristica().main()

# EOF
#############################################################################
