#!/usr/bin/env python
#############################################################################

from datetime import datetime
from PIL import Image

import argparse
import json
import numpy as np
import torch

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Define a simple neural network model
# It seems like GLSL get's maxed out anything 32x32 or greater
# Trying to go deep (giggetty)
class MLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        sizing = 16

        self.layers = torch.nn.Sequential(
            torch.nn.Linear(2, 16),  torch.nn.ReLU(),
            # note: 10 seems to be about the limit for 16x16
            torch.nn.Linear(16, 16), torch.nn.ReLU(),
            torch.nn.Linear(16, 16), torch.nn.ReLU(),
            torch.nn.Linear(16, 16), torch.nn.ReLU(),
            torch.nn.Linear(16, 16), torch.nn.ReLU(),
            torch.nn.Linear(16, 16), torch.nn.ReLU(),
            torch.nn.Linear(16, 16), torch.nn.ReLU(),
            torch.nn.Linear(16, 16), torch.nn.ReLU(),
            torch.nn.Linear(16, 16), torch.nn.ReLU(),
            torch.nn.Linear(16, 16), torch.nn.ReLU(),
            torch.nn.Linear(16, 16), torch.nn.ReLU(),
            torch.nn.Linear(16, 3), torch.nn.ReLU(),
        )

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

        #img_data = np.array(img) / 255.0
        # nu:
        #img_data = np.power(img_data, 2.2)  # Apply gamma correction

        h, w = img_data.shape[:2]

        x = np.linspace(-1, 1, w)
        y = np.linspace(-1, 1, h)
        xx, yy = np.meshgrid(x, y)
        coords = np.stack([xx, yy], axis=-1).reshape(-1, 2)  # (h*w, 2)
        colors = img_data.reshape(-1, 3)  # (h*w, 3)

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
        model = MLP().to(self.device)
        
        best_model = None
        best_loss = float('inf')

        # Define loss function and optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        loss_fn = torch.nn.MSELoss()

        # Training loop 
        epochs = 5000 * self.args.training
        Futuristica.LOG.info(f"Training for 5k x {self.args.training} (minutes) -> {epochs}");
        
        spammed = 0

        # don't rollback too often
        rollback_too_soon = 100 # TODO: parameterize this
        rollback_way_too_long = 5 * 10 * rollback_too_soon # TODO: parameterize this

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

            loss_v = loss.item()
            percent = (loss_v - best_loss) / best_loss * 100

            if loss_v < best_loss:
                spammed = spammed + 1
                if spammed < 4:
                    Futuristica.LOG.info(f"Epoch {epoch}: Checkpoint since is {loss_v:.6f} vs {best_loss:.6f}, {percent:.4f}%")
                best_loss = loss_v
                best_model = MLP()  # Create a new instance of the model
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
                    Futuristica.LOG.info(f"Epoch {epoch}: PUNT: {rollback_since_last} way is too long!")
                    has_checkpoint = True
                if rollback_since_last > rollback_too_soon and has_checkpoint:
                    Futuristica.LOG.warning(f"Epoch {epoch}: Rollback from {loss_v:.6f} to {best_loss:.6f}, {percent:.4f}%")
                    model.load_state_dict(best_model.state_dict())  # Load the previous best state dict
                    #model.cuda()  # Move the model back to GPU
                    model.to(self.device) # Move the model back to GPU
                    rollback_last = epoch
                    has_checkpoint = False # at least not recently...
                    spammed = 0
            else:
                spammed = 0
            
            if epoch % 1000 == 0:
                Futuristica.LOG.info(f"Epoch {epoch}: Loss = {loss_v:.6f} vs ({best_loss:.6f})")
                spammed = 0

        Futuristica.LOG.info("Training complete!")

        best_model.to(self.device)
        return best_model
    #end of train


    # this version is supposed to just keep the lowest lost model it sees
    # FIXME: untested! 
    def best_of_train(self, coords, colors):
        # Initialize model and move it to GPU
        model = MLP().to(self.device)

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

        return model
    #end of train

    # the og! this just does what it does!
    def og_train(self, coords, colors):
        # Initialize model and move it to GPU
        model = MLP().to(self.device)

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
        weights = {}
        for i, (name, param) in enumerate(model.named_parameters()):
            weights[name] = param.detach().cpu().numpy()  # Convert to NumPy
        np.savez(filename, **weights)
        Futuristica.LOG.info(f"Saved weights to {filename}")
    #end of export_weights


    def generate_image(self, model, filename = "generated_image.png", timestamp = False):
        size = self.args.size

        x_coords = np.linspace(-1, 1, size)
        y_coords = np.linspace(-1, 1, size)
        x_grid, y_grid = np.meshgrid(x_coords, y_coords)

        # Flatten the input coordinates
        inputs = np.stack((x_grid.flatten(), y_grid.flatten()), axis=1)

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

# end of class Futuristica

if __name__ == "__main__":
    Futuristica().main()

# EOF
#############################################################################
