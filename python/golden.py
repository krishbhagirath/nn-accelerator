# The ONE canonical integer implementation of the accelerator's arithmetic.
#
# The compiler uses it to calibrate shifts and produce expected outputs, and
# the cocotb tests / pipeline use it to check the RTL. If the RTL and this file
# ever disagree, one of them has a bug.
#
# Every function works on numpy int64 arrays. int64 is wide enough that no
# intermediate value can overflow, so these results are mathematically exact.

import json
from pathlib import Path

import numpy as np

INT8_MIN = -128
INT8_MAX = 127


def quantize_input(float_values, input_scale):
    """Float features -> INT8 activations (done in software, before the accelerator)."""
    q = np.round(np.asarray(float_values, dtype=np.float64) * input_scale)
    # Symmetric range: -127..127, so +x and -x quantize to +q and -q.
    return np.clip(q, -127, 127).astype(np.int64)


def fc_layer(inputs, weights, biases, shift, relu):
    """One fully-connected layer exactly as the RTL computes it.

    inputs:  [..., input_dim]        INT8 values
    weights: [output_dim, input_dim] INT8 values
    biases:  [output_dim]            INT32 values
    """
    # MAC stage: the accumulator starts at the bias, then adds input * weight.
    accumulator = inputs @ weights.T + biases

    # Requantize: arithmetic right shift (rounds toward -infinity), same as
    # SystemVerilog's >>> on a signed value.
    shifted = accumulator >> shift

    # Hidden layers: ReLU, so clamp to 0..127.
    # Final layer: signed logits, so clamp to -128..127.
    lowest = 0 if relu else INT8_MIN
    return np.clip(shifted, lowest, INT8_MAX)


def run_model(inputs, layers):
    """Run every layer in order. Returns the output of each layer."""
    layer_outputs = []
    activations = np.asarray(inputs, dtype=np.int64)
    for layer in layers:
        activations = fc_layer(
            activations,
            layer["weights"],
            layer["biases"],
            layer["output_shift"],
            layer["relu"],
        )
        layer_outputs.append(activations)
    return layer_outputs


def predict(final_outputs):
    """Predicted class = index of the largest output (first one wins on ties)."""
    return np.argmax(final_outputs, axis=-1)


# ------------------------------------------------------------
# Reading the compiler's output files
# ------------------------------------------------------------

def read_matrix(path):
    """Text file with one row of space-separated integers per line."""
    return np.loadtxt(path, dtype=np.int64, ndmin=2)


def read_vector(path):
    return np.loadtxt(path, dtype=np.int64, ndmin=1)


def load_compiled_model(compiled_dir):
    """Load manifest.json plus every layer's weights/biases as numpy arrays."""
    compiled_dir = Path(compiled_dir)
    manifest = json.loads((compiled_dir / "manifest.json").read_text())

    layers = []
    for layer_info in manifest["layers"]:
        layer = dict(layer_info)
        layer["weights"] = read_matrix(compiled_dir / layer_info["weights_file"])
        layer["biases"] = read_vector(compiled_dir / layer_info["biases_file"])
        layers.append(layer)

    test_inputs = read_matrix(compiled_dir / "test_inputs.txt")
    expected_outputs = read_matrix(compiled_dir / "expected_outputs.txt")
    test_labels = read_vector(compiled_dir / "test_labels.txt")

    return manifest, layers, test_inputs, expected_outputs, test_labels
