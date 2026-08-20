# Defines a tiny neural network whose first layer matches our RTL accelerator.
# The accelerator will eventually execute the 8 -> 16 layer + ReLU.
# The final 16 -> 2 classification layer will initially remain in Python.

import torch
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(42)

class TinyNN(nn.Module):

    def __init__(self):
        super().__init__()

        # 8 input features -> 16 neurons
        # This layer contains:
        #   16 * 8 = 128 weights
        #   16 biases
        #
        # This matches the dimensions of our hardware accelerator.
        self.fc1 = nn.Linear(8, 16)

        # Activation function already supported by our RTL post-processing
        self.relu = nn.ReLU()

        # Takes the 16 accelerator outputs and produces 2 class scores.
        # We will leave this layer in Python for now.
        self.fc2 = nn.Linear(16, 2)


    def forward(self, x):

        # This portion will eventually execute on our RTL accelerator.
        x = self.fc1(x)
        x = self.relu(x)

        # This portion currently executes in Python.
        x = self.fc2(x)

        return x



# ------------------------------------------------------------
# Create training data
# ------------------------------------------------------------

# Generate 2,000 samples.
# Each sample contains exactly 8 input features, matching our accelerator.
X = torch.randn(2000, 8)

# Create two classes based on a relationship between the 8 features.
#
# Class 1: weighted combination of features is positive
# Class 0: weighted combination is negative
#
# This gives the neural network a real pattern it must learn rather
# than assigning random labels.
score = (
    2.0 * X[:, 0]
    - 1.5 * X[:, 1]
    + 0.8 * X[:, 2]
    + 1.2 * X[:, 3]
    - 0.5 * X[:, 4]
    + 0.7 * X[:, 5]
    - 1.0 * X[:, 6]
    + 0.4 * X[:, 7]
)

y = (score > 0).long()


# ------------------------------------------------------------
# Split into training and test sets
# ------------------------------------------------------------

X_train = X[:1600]
y_train = y[:1600]

X_test = X[1600:]
y_test = y[1600:]


# ------------------------------------------------------------
# Create the model
# ------------------------------------------------------------

model = TinyNN()

# CrossEntropyLoss is used for classification.
criterion = nn.CrossEntropyLoss()

# Adam changes the model's weights/biases during training
# in order to reduce the classification error.
optimizer = optim.Adam(model.parameters(), lr=0.01)


# ------------------------------------------------------------
# Train the neural network
# ------------------------------------------------------------

num_epochs = 100

for epoch in range(num_epochs):

    # 1. Run all training samples through the neural network
    outputs = model(X_train)

    # 2. Compare predictions against the correct labels
    loss = criterion(outputs, y_train)

    # 3. Clear gradients calculated during the previous iteration
    optimizer.zero_grad()

    # 4. Calculate how each weight contributed to the error
    loss.backward()

    # 5. Update weights and biases to reduce the error
    optimizer.step()

    # Print progress every 10 epochs
    if (epoch + 1) % 10 == 0:
        print(
            f"Epoch {epoch + 1}/{num_epochs}, "
            f"Loss: {loss.item():.4f}"
        )


# ------------------------------------------------------------
# Test the trained model
# ------------------------------------------------------------

# We don't need gradients because we're no longer training.
with torch.no_grad():

    # Run the 400 unseen test samples through the model
    outputs = model(X_test)

    # outputs contains 2 scores per sample.
    # Pick whichever class has the larger score.
    predictions = torch.argmax(outputs, dim=1)

    # Calculate percentage of correct predictions
    accuracy = (predictions == y_test).float().mean()

    print(f"\nTest accuracy: {accuracy.item() * 100:.2f}%")



















# ------------------------------------------------------------
# Extract first-layer parameters
# ------------------------------------------------------------

fc1_weights = model.fc1.weight.detach()
fc1_biases = model.fc1.bias.detach()

print("\nFirst-layer weight shape:", fc1_weights.shape)
print("First-layer bias shape:", fc1_biases.shape)


# ------------------------------------------------------------
# Simple INT8 quantization
# ------------------------------------------------------------

# Find the largest absolute weight so we can scale all weights
# into the signed INT8 range [-127, 127].
max_weight = fc1_weights.abs().max()

weight_scale = 127.0 / max_weight

quantized_weights = torch.round(
    fc1_weights * weight_scale
).clamp(-127, 127).to(torch.int8)

# Biases need to be scaled consistently with the weighted sum.
quantized_biases = torch.round(
    fc1_biases * weight_scale
).to(torch.int32)


print("\nQuantized weights:")
print(quantized_weights)

print("\nQuantized biases:")
print(quantized_biases)






# ------------------------------------------------------------
# Quantize one real test sample for the RTL accelerator
# ------------------------------------------------------------

# Pick one sample from the test set
sample = X_test[0]

# Scale the input so the largest absolute value maps near 127
max_input = sample.abs().max()
input_scale = 127.0 / max_input

quantized_input = torch.round(
    sample * input_scale
).clamp(-127, 127).to(torch.int8)


# ------------------------------------------------------------
# Recompute bias scale for integer MAC arithmetic
# ------------------------------------------------------------

# The MAC multiplies:
#   quantized_input * quantized_weight
#
# Therefore the accumulator is scaled by:
#   input_scale * weight_scale
accumulator_scale = input_scale * weight_scale

quantized_biases = torch.round(
    fc1_biases * accumulator_scale
).to(torch.int32)


# ------------------------------------------------------------
# Quantized Python reference
# ------------------------------------------------------------

quantized_output = []

for neuron in range(16):

    total = int(quantized_biases[neuron])

    for i in range(8):
        total += (
            int(quantized_input[i])
            * int(quantized_weights[neuron][i])
        )

    # Requantize the 32-bit accumulator back toward INT8 range
    total = total >> 9

    # ReLU
    total = max(0, total)

    # Saturate
    total = min(127, total)

    quantized_output.append(total)


print("\nOriginal test sample:")
print(sample)

print("\nQuantized INT8 input:")
print(quantized_input)

print("\nQuantized INT32 biases:")
print(quantized_biases)

print("\nExpected RTL first-layer outputs:")
print(quantized_output)



# ------------------------------------------------------------
# Calibrate accumulator range
# ------------------------------------------------------------

max_acc = 0

for sample in X_train[:500]:
    max_input = sample.abs().max()
    input_scale = 127.0 / max_input

    q_input = torch.round(
        sample * input_scale
    ).clamp(-127, 127).to(torch.int8)

    acc_scale = input_scale * weight_scale

    q_biases = torch.round(
        fc1_biases * acc_scale
    ).to(torch.int32)

    for neuron in range(16):
        total = int(q_biases[neuron])

        for i in range(8):
            total += int(q_input[i]) * int(quantized_weights[neuron][i])

        max_acc = max(max_acc, abs(total))

print("\nMaximum observed accumulator:", max_acc)



## AUTO STORE WEIGHTS INTO TEXT FILES FOR RTL SIMULATION
# ------------------------------------------------------------
# Export quantized model data for RTL / Cocotb
# ------------------------------------------------------------

from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
model_data_dir = project_root / "model_data"
model_data_dir.mkdir(exist_ok=True)


def write_values(filename, values):
    """Write one integer value per line."""
    with open(model_data_dir / filename, "w") as f:
        for value in values:
            f.write(f"{int(value)}\n")


# Flatten 16x8 weights into 128 sequential values
flat_weights = quantized_weights.flatten()

write_values("weights.txt", flat_weights)
write_values("biases.txt", quantized_biases)
write_values("sample_input.txt", quantized_input)
write_values("expected_output.txt", quantized_output)

print(f"\nModel data exported to: {model_data_dir}")