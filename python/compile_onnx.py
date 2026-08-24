from pathlib import Path
import numpy as np
import onnx
from onnx import numpy_helper


# ------------------------------------------------------------
# Paths and accelerator hardware limits
# ------------------------------------------------------------

project_root = Path(__file__).resolve().parent.parent

onnx_path = project_root / "model_data" / "tiny_nn.onnx"
calibration_path = project_root / "model_data" / "calibration_inputs.txt"
output_dir = project_root / "compiled_model"

SUPPORTED_INPUTS = 8
SUPPORTED_NEURONS = 16


# ------------------------------------------------------------
# Load the ONNX model
# ------------------------------------------------------------

def load_model():

    model = onnx.load(onnx_path)

    return model


# ------------------------------------------------------------
# Read the model's stored weights and biases
# ------------------------------------------------------------

def get_parameters(model):

    parameters = {}

    for tensor in model.graph.initializer:

        # Convert ONNX tensor into a normal NumPy array
        array = numpy_helper.to_array(tensor)

        # Store it using its ONNX name
        parameters[tensor.name] = array

    return parameters


# ------------------------------------------------------------
# Find a linear layer that fits our accelerator
# ------------------------------------------------------------

def find_supported_layer(model, parameters):

    # Look through every operation in the neural network
    for node in model.graph.node:

        # Fully connected layers are represented as Gemm
        if node.op_type == "Gemm":

            # Gemm inputs:
            # input[0] = activations
            # input[1] = weights
            # input[2] = biases
            weight_name = node.input[1]
            bias_name = node.input[2]

            weights = parameters[weight_name]
            biases = parameters[bias_name]

            # Weight matrix tells us the layer dimensions
            output_neurons = weights.shape[0]
            input_features = weights.shape[1]

            print("\nChecking layer:")
            print("Input features:", input_features)
            print("Output neurons:", output_neurons)

            # Current accelerator supports an 8 -> 16 layer
            if (
                input_features == SUPPORTED_INPUTS
                and output_neurons == SUPPORTED_NEURONS
            ):
                print("SUPPORTED by accelerator")

                # Return this layer's parameters so they can
                # be quantized and mapped onto the hardware
                return weights, biases

            else:
                print("NOT supported by current accelerator")

    # No compatible layer was found
    return None, None


# ------------------------------------------------------------
# Calculate input scale from calibration data
# ------------------------------------------------------------

def get_input_scale():

    calibration_values = []

    with open(calibration_path, "r") as file:

        for line in file:

            if line.strip():
                calibration_values.append(
                    float(line.strip())
                )

    # Find the largest magnitude activation in the
    # calibration data so it maps into INT8 range
    max_input = max(
        abs(value)
        for value in calibration_values
    )

    input_scale = 127.0 / max_input

    return input_scale


# ------------------------------------------------------------
# Quantize the supported layer for the accelerator
# ------------------------------------------------------------

def quantize_layer(weights, biases, input_scale):

    # --------------------------------------------------------
    # Quantize weights to signed INT8
    # --------------------------------------------------------

    # Find the largest weight magnitude
    max_weight = np.max(np.abs(weights))

    # Scale weights into signed INT8 range
    weight_scale = 127.0 / max_weight

    quantized_weights = np.round(
        weights * weight_scale
    )

    # Keep values inside our INT8 range
    quantized_weights = np.clip(
        quantized_weights,
        -127,
        127
    ).astype(np.int8)


    # --------------------------------------------------------
    # Quantize biases to INT32
    # --------------------------------------------------------

    # The MAC computes:
    #
    # INT8 input * INT8 weight
    #
    # so the bias must use the same scale as the
    # INT32 MAC accumulator.
    accumulator_scale = input_scale * weight_scale

    quantized_biases = np.round(
        biases * accumulator_scale
    ).astype(np.int32)

    return quantized_weights, quantized_biases


# ------------------------------------------------------------
# Write values to a hardware data file
# ------------------------------------------------------------

def write_values(filename, values):

    with open(output_dir / filename, "w") as file:

        for value in values:

            # Store one integer per line
            file.write(f"{int(value)}\n")


# ------------------------------------------------------------
# Export hardware-ready model data
# ------------------------------------------------------------

def write_compiled_model(weights, biases):

    # Create compiled_model/ if it does not already exist
    output_dir.mkdir(exist_ok=True)

    # Hardware weight memory expects 128 sequential values,
    # so flatten the 16 x 8 matrix into one list.
    flat_weights = weights.flatten()

    write_values("weights.txt", flat_weights)
    write_values("biases.txt", biases)


# ------------------------------------------------------------
# Run the compiler
# ------------------------------------------------------------

model = load_model()

# Extract all stored ONNX parameters
parameters = get_parameters(model)

# Find a layer that can execute on our accelerator
supported_weights, supported_biases = find_supported_layer(
    model,
    parameters
)


# Make sure a compatible layer was actually found
if supported_weights is None:

    print("\nNo supported layer found.")

else:

    print("\nSelected accelerator layer:")
    print("Weight shape:", supported_weights.shape)
    print("Bias shape:", supported_biases.shape)

    print("\nFirst neuron's weights:")
    print(supported_weights[0])

    print("\nFirst bias:")
    print(supported_biases[0])


    # Calculate activation scale from calibration data
    input_scale = get_input_scale()

    print("\nCalculated input scale:", input_scale)


    # Convert FP32 model parameters into the integer
    # representation expected by our accelerator
    quantized_weights, quantized_biases = quantize_layer(
        supported_weights,
        supported_biases,
        input_scale
    )


    print("\nQuantized weight shape:", quantized_weights.shape)

    print("First neuron's quantized weights:")
    print(quantized_weights[0])

    print("\nFirst quantized bias:")
    print(quantized_biases[0])


    # Generate files that Cocotb/runtime can load
    # directly into accelerator memory
    write_compiled_model(
        quantized_weights,
        quantized_biases
    )

    print("\nCompiled model written to:", output_dir)