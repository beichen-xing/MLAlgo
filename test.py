import torch

# Load the trained model parameters
policy = torch.load("cnn_model.pth")
print(policy)  # This will show the state dictionary (weights and biases).
