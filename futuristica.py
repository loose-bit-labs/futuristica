#!/usr/bin/env python
#############################################################################

from datetime import datetime
from PIL import Image

import argparse
import json
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
    def __init__(self, coding=2):
        super().__init__()
        encoding_size = 2 + coding * 4 # 10 for coding=2
        layers = []
        layers.append(torch.nn.Linear(encoding_size, 16))
        layers.append(torch.nn.ReLU())
        for _ in range(10):
            layers.append(torch.nn.Linear(16, 16))
            layers.append(torch.nn.ReLU())
        layers.append(torch.nn.Linear(16, 3))
        # oops...layers.append(torch.nn.Sigmoid())
        layers.append(torch.nn.ReLU())
        self.layers = torch.nn.Sequential(*layers)


    def forward(self, x):
        return self.layers(x)


##
 #
 # Welcome to Futuristica
 #
 ##
class Futuristica:
    LOG = logging.getLogger(__name__)

    def __init__(self):
        self.parser = argparse.ArgumentParser(description="This is Futuristica")
        self.parser.add_argument("--image",     type=str)
        self.parser.add_argument("--size",      type=int, default=512)
        self.parser.add_argument("--weights",   type=str, default="weights.npz")
        self.parser.add_argument("--generated", type=str, default="generated_image.png")
        self.parser.add_argument("--training",  type=int, default=20)
        self.parser.add_argument("--coding",    type=int, default=2)
        self.parser.add_argument("--ckp",       type=str)
        self.parser.add_argument("--loss_fn",   choices=["pixel", "perceptual", "hybrid"], default="pixel")
    

    def main(self):
        self.args = self.parser.parse_args()
        Futuristica.LOG.info(f'settings: {json.dumps(vars(self.args))}')
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

    # this version supports rollback but can take longer
    def train_back(self, coords, colors):
        # Initialize model and move it to GPU
        model = Neuralistica(self.args.coding).to(self.device)

        if self.args.ckp:
            model = self.load_weights(model, self.args.ckp)
        
        best_model = None
        best_loss = float('inf')

        # Define loss function and optimizer

        loss_fns = {
            "pixel": torch.nn.MSELoss,
            "perceptual": self.perceptual_loser,
            "hybrid": self.hybrid_loser
        }
        if not self.args.loss_fn in loss_fns:
            Futuristica.LOG.warning(f"Unknown loss function '{self.args.loss_fn}', defaulting to 'pixel'");
            self.args.loss_fn = "pixel"
        loss_fn = loss_fns[self.args.loss_fn]()

        ################################################################################
        # set up the optimizer and scheduler

        #optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)
        ################################################################################


        # Training loop 
        epochs = 5000 * self.args.training
        Futuristica.LOG.info(f"Training for 5k x {self.args.training} (minutes) -> {epochs}");
        
        spammed = 0

        # don't rollback too often
        rollback_too_soon = 100 # TODO: parameterize this
        rollback_way_too_long = 10 * 10 * rollback_too_soon # TODO: parameterize this

        rollback_last = 0
        has_checkpoint = False

        imaged_last = 0
        imaged_too_soon = 100

        for epoch in range(epochs):
            optimizer.zero_grad()
            predictions = model(coords)  # Forward pass
            loss = loss_fn(predictions, colors)  # Compute loss
            loss.backward()  # Backpropagation
            optimizer.step()  # Update weights
            scheduler.step()  # Update learning rate

            loss_v = loss.item()
            percent = (loss_v - best_loss) / best_loss * 100

            now = f"Epoch {epoch}[{int(epoch * 100 / epochs)}%]"

            if loss_v < best_loss:
                spammed = spammed + 1
                if spammed < 4:
                    Futuristica.LOG.info(f"{now} Checkpoint since is {loss_v:.6f} vs {best_loss:.6f}, {percent:.4f}%")
                best_loss = loss_v
                best_model = Neuralistica(self.args.coding)  # Create a new instance of the model
                best_model.load_state_dict(model.state_dict())  # Copy the model's state dict
                best_model.cpu()  # Move the model to CPU
                has_checkpoint = True
                if epoch - imaged_last > imaged_too_soon:
                    self.generate_image(model, self.args.generated, True)
                    self.export_weights(model, self.args.weights) # warum nicht?
                    imaged_last = epoch
            elif loss_v > best_loss:
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
                Futuristica.LOG.info(f"{now}: Loss: {loss_v:.6f} vs ({best_loss:.6f}), LR:{lr:.6e}")
                spammed = 0

        Futuristica.LOG.info("Training complete!")

        best_model.to(self.device)
        return best_model
    #end of train


    # this version is supposed to just keep the lowest lost model it sees
    # FIXME: untested! 
    def best_of_train(self, coords, colors):
        # Initialize model and move it to GPU
        model = Neuralistica(self.args.coding).to(self.device)

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
        model = Neuralistica(self.args.coding).to(self.device)

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
        for name, param in model.named_parameters():
            param.data = torch.from_numpy(weights[name]).to(param.device)
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

        # Convert output to image
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


    # this seems to help learn sharp edges, but creates artifacts too
    def positional_encoding(self, coords, L=10):
        """Encodes (x, y) coordinates into a high-dimensional space"""
        encodings = [coords]
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
    def og_positional_encoding(self, coords, L=10):
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



# end of class Futuristica

if __name__ == "__main__":
    Futuristica().main()

# EOF
#############################################################################
