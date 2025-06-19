# Defines the model. Bloated because it includes optional things from testing
import torch
import torch.nn as nn
import torch.nn.functional as F
from EndLayers import EndLayers
import config
from utils import ACI_CATEGORY_MAPPING

class Config:
    """Configuration settings for the model."""
    parameters = {
        "Activation": ["Leaky"],  # Choose activation: Leaky, Sigmoid, Tanh, etc.
        "Nodes": [128],           # Number of nodes in the fully connected layers
        "Dropout": [0.5],         # Dropout rate
        "CLASSES": [11],          # Number of classes
        "threshold": 0.25,       # Cutoff for end layers (if applicable)
        "Number of Layers": [0]   # Additional layers post-FC1
    }
    dataparallel = False         # Enable DataParallel if required


class Compact1DCNN(nn.Module):
    def __init__(self, num_features, num_classes):
        """
        Args:
            num_features (int): Number of input features.
            num_classes (int): Number of output classes.
        """
        super(Compact1DCNN, self).__init__()
        # Pooling factors and convolutional channels (fixed for now)
        self.maxpooling = [4, 2]
        self.conv_channels = [32, 64]

        # Convolutional layers
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=self.conv_channels[0],
                               kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(in_channels=self.conv_channels[0], out_channels=self.conv_channels[1],
                               kernel_size=3, stride=1, padding=1)

        # Calculate the input size for the fully connected layers
        fc_input_length = num_features // self.maxpooling[0] // self.maxpooling[1]
        self.fc_input_size = fc_input_length * self.conv_channels[-1]

        # Activation function from config
        activations = {
            "Sigmoid": nn.Sigmoid(),
            "Tanh": nn.Tanh(),
            "Leaky": nn.LeakyReLU(),
            "Elu": nn.ELU(),
            "PRElu": nn.PReLU(),
            "Softplus": nn.Softplus(),
            "Softmax": nn.Softmax(dim=1)
        }
        act_choice = Config.parameters["Activation"][0]  # e.g., "Leaky"
        self.activation = activations.get(act_choice, nn.ReLU())

        # Fully connected layers
        nodes = Config.parameters["Nodes"][0]
        self.fc1 = nn.Linear(self.fc_input_size, nodes)
        self.fc2 = nn.Linear(nodes, num_classes)

        # Additional (configurable) layers after fc1
        self.addedLayers = nn.Sequential()
        for _ in range(Config.parameters["Number of Layers"][0]):
            self.addedLayers.append(nn.Linear(nodes, nodes))
            # Use the configured activation function
            self.addedLayers.append(self.activation)

        # Dropout layer from config
        self.dropout = nn.Dropout(Config.parameters["Dropout"][0])

        # Flattening before FC layers
        self.flatten = nn.Flatten()

        # Build the sequential package that processes features from the CNN into logits
        self.sequencePackage = nn.Sequential(
            self.flatten,
            self.fc1,
            self.activation,
            self.addedLayers,
            self.dropout,
            self.fc2
        )
        
        self.end = EndLayers(num_classes=num_classes, cutoff=Config.parameters["threshold"])

        # Class weights (as in the full model, where benign is weighted more)
        self.weights = torch.ones(num_classes)
        self.weights[0] += 1  # Increase weight for class 0 (e.g., "Benign")

        # Additional attributes used during training/evaluation
        self.batchnum = 0
        self.batch_fdHook = None

        # Store category mapping (this doesn't change)
        self.class_to_category_mapping = ACI_CATEGORY_MAPPING

    def forward(self, x):
        """
        Forward pass:
          - Applies two convolutional layers (each with activation and pooling).
          - Dynamically adjusts output weights based on predictions.
          - Feeds the result through the sequential FC package.
          - Optionally applies additional processing via `self.end`.
        
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, num_features].
        
        Returns:
            torch.Tensor: Output logits of shape [batch_size, num_classes].
        """
        # Ensure input is float and add a channel dimension.
        x = x.float()
        x = x.unsqueeze(1)  # Shape: [batch_size, 1, num_features]

        # Convolutional layers with activation and pooling.
        x = self.activation(self.conv1(x))
        x = F.max_pool1d(x, kernel_size=self.maxpooling[0])
        x = self.activation(self.conv2(x))
        x = F.max_pool1d(x, kernel_size=self.maxpooling[1])

        # Pass through the fully connected sequential package.
        x = self.sequencePackage[:-1](x)  # Exclude final layer for now

        # Compute logits
        logits = self.fc2(x)

        # Get current config values
        use_tta = config.USE_TTA
        weight_shift_factor = config.WEIGHT_SHIFT_FACTOR
        logit_shift_factor = config.LOGIT_SHIFT_FACTOR
        normalize_weights = config.NORMALIZE_WEIGHTS

        if use_tta or logit_shift_factor != 0:
            # Define phase transitions for categories
            phase_transitions = {
                "Benign": ["Benign", "Reckoning"],
                "Reckoning": ["Reckoning", "DoS"],
                "DoS": ["DoS", "Brute Force"],
                "Brute Force": ["Brute Force"]
            }

            # Get predictions
            predictions = torch.argmax(logits, dim=1)
            
            # Create a mask for logit adjustments
            logit_adjustments = torch.zeros_like(logits)
            
            # Calculate weight adjustments
            weight_adjustments = torch.zeros_like(self.fc2.weight)
            
            with torch.no_grad():
                for pred in predictions:
                    pred_idx = pred.item()
                    if pred_idx >= len(self.class_to_category_mapping):
                        continue
                        
                    current_category = self.class_to_category_mapping[pred_idx]
                    target_categories = phase_transitions.get(current_category, [])

                    # Calculate adjustments for target categories
                    for target_category in target_categories:
                        target_classes = [
                            cls_idx for cls_idx, cat in self.class_to_category_mapping.items()
                            if cat == target_category and cls_idx < self.fc2.weight.size(0)
                        ]
                        for target_class in target_classes:
                            if weight_shift_factor != 0:
                                weight_adjustments[target_class] += weight_shift_factor
                            
                            # Apply logit shift to target classes
                            logit_adjustments[:, target_class] += logit_shift_factor

            # Apply weight adjustments
            if weight_shift_factor != 0:
                adjusted_weights = self.fc2.weight + weight_adjustments
                if normalize_weights:
                    # Normalize each weight vector
                    norms = torch.norm(adjusted_weights, dim=1, keepdim=True)
                    norms[norms == 0] = 1  # Avoid division by zero
                    adjusted_weights = adjusted_weights / norms
                
                # Use adjusted weights for final logits
                logits = F.linear(x, adjusted_weights, self.fc2.bias)

            # Apply logit adjustments
            logits = logits + logit_adjustments

        if config.END_LAYER:
            logits = self.end(logits)

        return logits