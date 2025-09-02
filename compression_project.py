# ===========================
# MODEL COMPRESSION PROJECT - MLP + MNIST
# ===========================
# This script trains an MLP model on MNIST, applies pruning & quantization,
# exports to ONNX, quantizes again with ONNX Runtime, runs inference,
# and compares accuracy, size, and runtime.
# ===========================

import torch                   # PyTorch main library
import torch.nn as nn          # Neural network layers
import torch.optim as optim    # Optimizers (SGD, Adam, etc.)
import torch.nn.functional as F # Activation functions & loss functions
from torchvision import datasets, transforms  # Dataset loading & preprocessing
import torch.nn.utils.prune as prune  # Pruning utilities
import onnxruntime as ort      # ONNX Runtime for inference
import numpy as np             # NumPy for array operations
import time                    # For measuring inference time
import os                      # For checking file sizes
from onnxruntime.quantization import quantize_dynamic, QuantType  # ONNX quantization

# ---------------------------
# STEP 1 – LOAD DATASET
# ---------------------------
transform = transforms.Compose([transforms.ToTensor()])  # Convert images to tensors

# Download and load the MNIST training dataset
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
# Load the MNIST test dataset (download=False since it's already downloaded above)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform)

# Create DataLoaders to batch and shuffle data
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)

# ---------------------------
# STEP 2 – DEFINE MLP MODEL
# ---------------------------
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        # Fully connected layer: input (28x28 pixels) → 256 neurons
        self.fc1 = nn.Linear(28*28, 256)
        # Fully connected layer: 256 → 128 neurons
        self.fc2 = nn.Linear(256, 128)
        # Fully connected layer: 128 → 10 output classes (digits 0–9)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)  # Flatten the image into a vector
        x = F.relu(self.fc1(x))  # Apply ReLU after first layer
        x = F.relu(self.fc2(x))  # Apply ReLU after second layer
        x = self.fc3(x)          # Output layer (no activation, raw logits)
        return x

# Create model instance
model = MLP()

# ---------------------------
# STEP 3 – TRAIN BASELINE
# ---------------------------
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam optimizer

# Training function
def train(model, train_loader, optimizer):
    model.train()  # Set model to training mode
    for data, target in train_loader:
        optimizer.zero_grad()             # Clear gradients
        output = model(data)               # Forward pass
        loss = F.cross_entropy(output, target)  # Compute cross-entropy loss
        loss.backward()                    # Backpropagation
        optimizer.step()                   # Update weights

# Testing/validation function
def test(model, test_loader):
    model.eval()  # Set model to evaluation mode
    correct = 0
    with torch.no_grad():  # No gradient calculation during testing
        for data, target in test_loader:
            output = model(data)           # Forward pass
            pred = output.argmax(dim=1)    # Get class with highest score
            correct += pred.eq(target).sum().item()  # Count correct predictions
    return 100. * correct / len(test_loader.dataset)  # Return accuracy %

# Train model for 3 epochs and print accuracy
for epoch in range(3):
    train(model, train_loader, optimizer)
    acc = test(model, test_loader)
    print(f"Epoch {epoch+1} - Accuracy: {acc:.2f}%")

# Save baseline accuracy
baseline_acc = test(model, test_loader)
# Save the trained model to disk
torch.save(model.state_dict(), "baseline_model.pt")

# ---------------------------
# STEP 4 – APPLY PRUNING
# ---------------------------
# Apply L1 unstructured pruning to remove 30% of weights in each layer
prune.l1_unstructured(model.fc1, name='weight', amount=0.3)
prune.l1_unstructured(model.fc2, name='weight', amount=0.3)
prune.l1_unstructured(model.fc3, name='weight', amount=0.3)

# Remove pruning masks so pruned weights are permanently removed
prune.remove(model.fc1, 'weight')
prune.remove(model.fc2, 'weight')
prune.remove(model.fc3, 'weight')

# Test model after pruning
pruned_acc = test(model, test_loader)
print(f"After pruning - Accuracy: {pruned_acc:.2f}%")

# ---------------------------
# STEP 5 – PYTORCH QUANTIZATION
# ---------------------------
# Apply dynamic quantization (convert weights to int8)
quantized_model = torch.quantization.quantize_dynamic(
    model, {nn.Linear}, dtype=torch.qint8
)
# Test model after pruning + quantization
quantized_acc = test(quantized_model, test_loader)
print(f"After pruning + PyTorch quantization - Accuracy: {quantized_acc:.2f}%")

# ---------------------------
# STEP 6 – EXPORT TO ONNX (FP32 PRUNED)
# ---------------------------
dummy_input = torch.randn(1, 1, 28, 28)  # Fake input to trace the model
torch.onnx.export(
    model, dummy_input, "model.onnx",  # Export to model.onnx
    export_params=True, opset_version=12, do_constant_folding=True,  # Optimization
    input_names=['input'], output_names=['output'],  # Naming inputs/outputs
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}  # Flexible batch size
)

# ---------------------------
# STEP 7 – ONNX QUANTIZATION
# ---------------------------
# Quantize ONNX model weights to int8
quantize_dynamic("model.onnx", "model_quant.onnx", weight_type=QuantType.QInt8)
print("ONNX quantized model saved as model_quant.onnx")

# ---------------------------
# STEP 8 – INFERENCE WITH ONNX RUNTIME
# ---------------------------
# Function to run inference using ONNX Runtime and measure speed
def onnx_inference(model_path):
    ort_session = ort.InferenceSession(model_path)  # Load ONNX model
    correct = 0
    start_time = time.time()
    for data, target in test_loader:
        ort_inputs = {ort_session.get_inputs()[0].name: data.numpy()}  # Prepare input
        ort_outs = ort_session.run(None, ort_inputs)                   # Run inference
        preds = np.argmax(ort_outs[0], axis=1)                         # Get predicted class
        correct += (preds == target.numpy()).sum()                     # Count correct predictions
    acc = 100. * correct / len(test_loader.dataset)                    # Accuracy %
    avg_time = (time.time() - start_time) * 1000 / len(test_loader.dataset)  # Time per sample (ms)
    return acc, avg_time

# Run ONNX FP32 model
onnx_acc, onnx_time = onnx_inference("model.onnx")
# Run ONNX INT8 quantized model
onnx_quant_acc, onnx_quant_time = onnx_inference("model_quant.onnx")

# ---------------------------
# STEP 10 – TFLITE INFERENCE
# ---------------------------


def tflite_inference(model_path):
    """
    Run inference using TensorFlow Lite model and return accuracy + avg time.
    Processes one sample at a time since TFLite expects batch size = 1.
    """
    import tensorflow as tf

    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    correct = 0
    start_time = time.time()

    for data, target in test_loader:
        # Loop through each sample in the batch
        for i in range(len(data)):
            # Pick a single image and keep batch dimension = 1
            input_data = data[i].unsqueeze(0).numpy().astype(np.float32)  # shape [1, 1, 28, 28]

            # Set input tensor
            interpreter.set_tensor(input_details[0]['index'], input_data)

            # Run inference
            interpreter.invoke()

            # Get prediction
            output_data = interpreter.get_tensor(output_details[0]['index'])
            pred = np.argmax(output_data, axis=1)

            # Compare with true label
            correct += int(pred[0] == target[i].item())

    acc = 100. * correct / len(test_loader.dataset)
    avg_time = (time.time() - start_time) * 1000 / len(test_loader.dataset)  # ms per sample
    return acc, avg_time


# STEP 9 – RESULTS TABLE
# ---------------------------
# Get sizes of saved models
size_baseline = os.path.getsize("baseline_model.pt") / 1e6
size_onnx = os.path.getsize("model.onnx") / 1e6
size_onnx_quant = os.path.getsize("model_quant.onnx") / 1e6

# ---------------------------
# STEP 11 – RUN TFLITE MODEL
# ---------------------------
tflite_acc, tflite_time = tflite_inference("model.tflite")
size_tflite = os.path.getsize("model.tflite") / 1e6


# Print comparison table
print("\n===== FINAL RESULTS =====")
print(f"{'Model':<35} {'Size (MB)':<12} {'Accuracy (%)':<15} {'Time (ms/sample)':<20}")
print("-" * 90)
print(f"{'PyTorch Baseline FP32':<35} {size_baseline:<12.2f} {baseline_acc:<15.2f} {'N/A':<20}")
print(f"{'PyTorch Pruned FP32':<35} {size_baseline:<12.2f} {pruned_acc:<15.2f} {'N/A':<20}")
print(f"{'PyTorch Pruned+Quant INT8':<35} {size_baseline:<12.2f} {quantized_acc:<15.2f} {'N/A':<20}")
print(f"{'ONNX Pruned FP32':<35} {size_onnx:<12.2f} {onnx_acc:<15.2f} {onnx_time:<20.4f}")
print(f"{'ONNX Pruned+Quant INT8':<35} {size_onnx_quant:<12.2f} {onnx_quant_acc:<15.2f} {onnx_quant_time:<20.4f}")


print(f"{'TFLite Pruned+Quant INT8':<35} {size_tflite:<12.2f} {tflite_acc:<15.2f} {tflite_time:<20.4f}")

# ===========================
# EXTRA ANALYSIS – REPORTS & GRAPHS
# ===========================
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def detailed_evaluation(model, test_loader):
    model.eval()
    all_preds, all_targets = [], []
    
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            preds = output.argmax(dim=1)
            all_preds.extend(preds.numpy())
            all_targets.extend(target.numpy())
    
    # Accuracy
    acc = 100. * np.sum(np.array(all_preds) == np.array(all_targets)) / len(all_targets)
    print(f"\nFinal Detailed Evaluation - Accuracy: {acc:.2f}%")
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(all_targets, all_preds, digits=4))
    
    # Confusion matrix plot
    cm = confusion_matrix(all_targets, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=range(10))
    disp.plot(cmap=plt.cm.Blues, values_format='d')
    plt.title("Confusion Matrix (MNIST Digits)")
    plt.show()

# Run detailed evaluation on the final PyTorch baseline model
detailed_evaluation(model, test_loader)
