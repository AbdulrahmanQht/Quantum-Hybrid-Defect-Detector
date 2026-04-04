import torch

print("--- Hybrid QNN CPU Layer Keys ---")
# Load the saved weights
checkpoint = torch.load("models/qnn_cpu.pth", map_location="cpu")

# 1. See the names of all layers
print(checkpoint.keys())

# 2. Inspect the weights of the fusion classifier
# In your code, this is likely 'classifier.0.weight'
weights = checkpoint["classifier.0.weight"]

# 3. Check the magnitude of weights
print(f"Mean weight value: {weights.abs().mean()}")

print("\n")
# Load the classical CNN weights
# Make sure the path matches where cnn.py saved the file
path = "models/cnn_model.pth"
checkpoint = torch.load(path, map_location="cpu")

# 1. List all layers (State Dict Keys)
print("--- Classical CNN Layer Keys ---")
print(checkpoint.keys())

# 2. Check the "strength" of the feature extraction
# 'model.0.weight' is your first 7x7 Convolutional layer
first_layer_weights = checkpoint["model.0.weight"]
print(
    f"\nFirst Conv Layer Mean Absolute Weight: {first_layer_weights.abs().mean().item():.5f}"
)

# 3. Check the final classification layer
# Based on cnn.py, the last linear layer is likely 'model.13.weight'
last_layer_weights = checkpoint["model.13.weight"]
print(
    f"Final Classifier Mean Absolute Weight: {last_layer_weights.abs().mean().item():.5f}"
)
