# The model compiler: ONNX (float) -> accelerator-ready INT8 files.
#
# This is the single source of truth for quantization. It produces:
#
#   compiled/<model>/
#       manifest.json            layer sizes, ReLU flags, output shifts
#       layer_<i>_weights.txt    one row per output neuron, INT8
#       layer_<i>_biases.txt     one INT32 per line
#       test_inputs.txt          quantized INT8 test inputs, one sample per line
#       expected_outputs.txt     golden final-layer INT8 outputs, one per line
#       test_labels.txt          true class of each test sample
#
# Quantization scheme (deliberately simple):
#   - weights:     symmetric per-tensor INT8,  w_q = round(w * 127 / max|w|)
#   - activations: INT8, input scale from calibration data
#   - biases:      INT32 at the accumulator's scale (input_scale * weight_scale)
#   - requantize:  arithmetic right shift by output_shift (a power-of-two scale)
#
# Usage:  python python/compile_onnx.py --model digits

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper

import golden

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
COMPILED_DIR = PROJECT_ROOT / "compiled"

# Must match the RTL parameters in rtl/nn_accelerator_top.sv.
MAX_INPUTS = 64
MAX_OUTPUTS = 32
MAX_SHIFT = 31  # the shift is a 5-bit field in hardware


# ------------------------------------------------------------
# Step 1: read the float layers out of the ONNX graph
# ------------------------------------------------------------

def read_onnx_layers(onnx_path):
    """Walk the graph and turn it into a list of fully-connected layers.

    Supported pattern: Gemm [Relu] Gemm [Relu] ... Anything else is rejected,
    because the hardware cannot run it.
    """
    model = onnx.load(onnx_path)
    onnx.checker.check_model(model)
    params = {t.name: numpy_helper.to_array(t) for t in model.graph.initializer}

    layers = []
    for node in model.graph.node:
        if node.op_type == "Gemm":
            attrs = {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}
            if attrs.get("transA", 0) != 0:
                raise ValueError("Gemm with transA=1 is not supported")
            if attrs.get("alpha", 1.0) != 1.0 or attrs.get("beta", 1.0) != 1.0:
                raise ValueError("Gemm with alpha/beta != 1 is not supported")

            weights = params[node.input[1]].astype(np.float64)
            # Gemm computes x @ B (+ C). With transB=1, B is stored [out, in],
            # which is the layout we want. Otherwise transpose it.
            if attrs.get("transB", 0) == 0:
                weights = weights.T
            biases = params[node.input[2]].astype(np.float64)
            layers.append({"weights": weights, "biases": biases, "relu": False})

        elif node.op_type == "Relu":
            if not layers:
                raise ValueError("Relu before any Gemm is not supported")
            layers[-1]["relu"] = True

        else:
            raise ValueError(f"ONNX op '{node.op_type}' is not supported by the accelerator")

    for i, layer in enumerate(layers):
        out_dim, in_dim = layer["weights"].shape
        if in_dim > MAX_INPUTS or out_dim > MAX_OUTPUTS:
            raise ValueError(f"layer {i} is {in_dim}->{out_dim}, hardware max is "
                             f"{MAX_INPUTS}->{MAX_OUTPUTS}")
    return layers


# ------------------------------------------------------------
# Step 2: quantize
# ------------------------------------------------------------

def choose_output_shift(accumulators, relu):
    """Smallest right shift that brings the largest calibration value into range.

    Smallest = keep as many bits of precision as possible.
    Large enough = the biggest value seen on calibration data does not saturate.
    For ReLU layers only positive values survive, so only those matter.
    """
    if relu:
        peak = int(max(accumulators.max(), 0))
    else:
        peak = int(np.abs(accumulators).max())

    shift = 0
    while (peak >> shift) > golden.INT8_MAX and shift < MAX_SHIFT:
        shift += 1
    return shift


def quantize_model(float_layers, calibration_inputs):
    # Input scale: map the largest calibration magnitude to 127.
    input_scale = 127.0 / np.abs(calibration_inputs).max()

    # activation_scale = how many integer steps one float unit is worth, for
    # the activations ENTERING the current layer.
    activation_scale = input_scale
    activations = golden.quantize_input(calibration_inputs, input_scale)

    quantized_layers = []
    for layer in float_layers:
        weights = layer["weights"]
        weight_scale = 127.0 / np.abs(weights).max()
        q_weights = np.clip(np.round(weights * weight_scale), -127, 127).astype(np.int64)

        # (input * input_scale) * (weight * weight_scale) lands at this scale,
        # so the bias must be expressed at the same scale to be added to it.
        accumulator_scale = activation_scale * weight_scale
        q_biases = np.round(layer["biases"] * accumulator_scale).astype(np.int64)

        # The RTL accumulator is 32 bits. Check the worst case can't overflow.
        in_dim = weights.shape[1]
        worst_case = in_dim * 128 * 128 + np.abs(q_biases).max()
        assert worst_case < 2**31, "accumulator could overflow INT32"

        # Run calibration data through THIS layer to pick its shift.
        accumulators = activations @ q_weights.T + q_biases
        shift = choose_output_shift(accumulators, layer["relu"])

        quantized_layers.append({
            "weights": q_weights,
            "biases": q_biases,
            "relu": layer["relu"],
            "output_shift": shift,
            "weight_scale": weight_scale,
        })

        # Feed the next layer with exactly what the hardware would produce.
        activations = golden.fc_layer(activations, q_weights, q_biases, shift, layer["relu"])
        activation_scale = accumulator_scale / (2 ** shift)

    return input_scale, quantized_layers


# ------------------------------------------------------------
# Step 3: write the files
# ------------------------------------------------------------

def write_compiled_model(name, input_scale, layers, test_inputs_float, test_labels):
    out_dir = COMPILED_DIR / name
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    manifest_layers = []
    for i, layer in enumerate(layers):
        weights_file = f"layer_{i}_weights.txt"
        biases_file = f"layer_{i}_biases.txt"
        np.savetxt(out_dir / weights_file, layer["weights"], fmt="%d")
        np.savetxt(out_dir / biases_file, layer["biases"], fmt="%d")
        out_dim, in_dim = layer["weights"].shape
        manifest_layers.append({
            "input_dim": int(in_dim),
            "output_dim": int(out_dim),
            "relu": bool(layer["relu"]),
            "output_shift": int(layer["output_shift"]),
            "weights_file": weights_file,
            "biases_file": biases_file,
        })

    # Test data: quantize inputs once, then compute golden outputs.
    q_test_inputs = golden.quantize_input(test_inputs_float, input_scale)
    golden_layers = [dict(m, weights=l["weights"], biases=l["biases"])
                     for m, l in zip(manifest_layers, layers)]
    expected = golden.run_model(q_test_inputs, golden_layers)[-1]

    np.savetxt(out_dir / "test_inputs.txt", q_test_inputs, fmt="%d")
    np.savetxt(out_dir / "expected_outputs.txt", expected, fmt="%d")
    np.savetxt(out_dir / "test_labels.txt", test_labels, fmt="%d")

    manifest = {
        "model": name,
        "input_dim": manifest_layers[0]["input_dim"],
        "input_scale": float(input_scale),
        "num_layers": len(manifest_layers),
        "num_test_samples": int(len(q_test_inputs)),
        "layers": manifest_layers,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=4))
    return manifest


def compile_model(name):
    model_dir = MODELS_DIR / name
    float_layers = read_onnx_layers(model_dir / "model.onnx")
    calibration = np.loadtxt(model_dir / "calibration_inputs.txt", ndmin=2)
    test_inputs = np.loadtxt(model_dir / "test_inputs.txt", ndmin=2)
    test_labels = np.loadtxt(model_dir / "test_labels.txt", dtype=np.int64, ndmin=1)

    input_scale, layers = quantize_model(float_layers, calibration)
    manifest = write_compiled_model(name, input_scale, layers, test_inputs, test_labels)

    shape = " -> ".join([str(manifest["input_dim"])] +
                        [str(l["output_dim"]) for l in manifest["layers"]])
    shifts = [l["output_shift"] for l in manifest["layers"]]
    print(f"[compile] {name:12s} {shape:16s} input_scale={input_scale:.3f} shifts={shifts}")
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="all")
    args = parser.parse_args()
    if args.model == "all":
        names = sorted(p.name for p in MODELS_DIR.iterdir() if (p / "model.onnx").exists())
    else:
        names = [args.model]
    for name in names:
        compile_model(name)


if __name__ == "__main__":
    main()
