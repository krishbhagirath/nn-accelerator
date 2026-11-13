# Plain Python unit tests (no simulator): golden arithmetic, compiler, ONNX.
#
# Run:  python -m pytest -q tests

import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime
import pytest
from onnx import TensorProto, helper, numpy_helper

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "python"))
import compile_onnx  # noqa: E402
import golden  # noqa: E402

MODELS = ["tiny", "iris", "cancer", "digits", "digits_deep"]


# ------------------------------------------------------------
# Golden integer arithmetic
# ------------------------------------------------------------

def test_shift_rounds_toward_minus_infinity():
    # Same as SystemVerilog >>> on a signed value
    w = np.array([[1]])
    for bias, shift, expected in [(-1, 1, -1), (-3, 1, -2), (3, 1, 1), (-256, 8, -1), (255, 8, 0)]:
        assert golden.fc_layer(np.array([0]), w, np.array([bias]), shift, False)[0] == expected


def test_clamping():
    w = np.array([[1]])
    big, small = np.array([1000]), np.array([-1000])
    assert golden.fc_layer(np.array([0]), w, big, 0, False)[0] == 127
    assert golden.fc_layer(np.array([0]), w, small, 0, False)[0] == -128
    assert golden.fc_layer(np.array([0]), w, small, 0, True)[0] == 0
    assert golden.fc_layer(np.array([0]), w, big, 0, True)[0] == 127


def test_fc_layer_matches_hand_computation():
    x = np.array([2, -1, 3])
    w = np.array([[1, 2, 3], [-4, 5, -6]])
    b = np.array([10, 0])
    # neuron 0: 2 - 2 + 9 + 10 = 19 ; neuron 1: -8 - 5 - 18 = -31
    assert golden.fc_layer(x, w, b, 0, False).tolist() == [19, -31]
    assert golden.fc_layer(x, w, b, 2, True).tolist() == [4, 0]


def test_quantize_input_is_symmetric():
    q = golden.quantize_input(np.array([-2.0, -1.0, 0.0, 1.0, 2.0]), 63.5)
    assert q.tolist() == [-127, -64, 0, 64, 127]


# ------------------------------------------------------------
# Compiler
# ------------------------------------------------------------

def test_choose_output_shift():
    assert compile_onnx.choose_output_shift(np.array([127]), relu=True) == 0
    assert compile_onnx.choose_output_shift(np.array([128]), relu=True) == 1
    assert compile_onnx.choose_output_shift(np.array([1000, -50000]), relu=True) == 3
    assert compile_onnx.choose_output_shift(np.array([1000, -50000]), relu=False) == 9
    # the chosen shift always brings the peak into range, and one less would not
    for peak in [200, 4097, 123456]:
        s = compile_onnx.choose_output_shift(np.array([peak]), relu=False)
        assert (peak >> s) <= 127 < (peak >> (s - 1))


def make_onnx(path, ops):
    """Build a tiny ONNX graph: Gemm followed by the given op types."""
    w = numpy_helper.from_array(np.ones((2, 3), dtype=np.float32), "w")
    b = numpy_helper.from_array(np.zeros(2, dtype=np.float32), "b")
    nodes = [helper.make_node("Gemm", ["x", "w", "b"], ["y0"], transB=1)]
    for i, op in enumerate(ops):
        nodes.append(helper.make_node(op, [f"y{i}"], [f"y{i + 1}"]))
    graph = helper.make_graph(
        nodes, "g",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3])],
        [helper.make_tensor_value_info(f"y{len(ops)}", TensorProto.FLOAT, [1, 2])],
        [w, b])
    onnx.save(helper.make_model(graph), path)


def test_compiler_reads_gemm_relu(tmp_path):
    make_onnx(tmp_path / "m.onnx", ["Relu"])
    layers = compile_onnx.read_onnx_layers(tmp_path / "m.onnx")
    assert len(layers) == 1 and layers[0]["relu"] and layers[0]["weights"].shape == (2, 3)


def test_compiler_rejects_unsupported_ops(tmp_path):
    make_onnx(tmp_path / "m.onnx", ["Sigmoid"])
    with pytest.raises(ValueError, match="Sigmoid"):
        compile_onnx.read_onnx_layers(tmp_path / "m.onnx")


# ------------------------------------------------------------
# The real trained / compiled models
# ------------------------------------------------------------

@pytest.mark.parametrize("model", MODELS)
def test_onnx_model_is_valid(model):
    path = PROJECT_ROOT / "models" / model / "model.onnx"
    graph = onnx.load(path)
    onnx.checker.check_model(graph)
    assert {n.op_type for n in graph.graph.node} <= {"Gemm", "Relu"}


@pytest.mark.parametrize("model", MODELS)
def test_onnxruntime_matches_numpy_float_forward(model):
    """Our reading of the ONNX graph (weights, transposes, ReLUs) is correct."""
    path = PROJECT_ROOT / "models" / model / "model.onnx"
    layers = compile_onnx.read_onnx_layers(path)
    x = np.loadtxt(PROJECT_ROOT / "models" / model / "test_inputs.txt", ndmin=2, dtype=np.float32)[:5]
    session = onnxruntime.InferenceSession(str(path))
    for sample in x:
        h = sample.astype(np.float64)
        for layer in layers:
            h = layer["weights"] @ h + layer["biases"]
            if layer["relu"]:
                h = np.maximum(h, 0)
        ort = session.run(None, {"input": sample[None, :]})[0][0]
        assert np.allclose(h, ort, atol=1e-4)


@pytest.mark.parametrize("model", MODELS)
def test_compiled_model_is_consistent(model):
    """expected_outputs.txt really is the golden model's output, and fits hardware limits."""
    manifest, layers, inputs, expected, labels = golden.load_compiled_model(
        PROJECT_ROOT / "compiled" / model)
    assert manifest["num_layers"] == len(layers) <= compile_onnx.MAX_LAYERS
    assert len(inputs) == len(expected) == len(labels) == manifest["num_test_samples"]
    assert np.array_equal(golden.run_model(inputs, layers)[-1], expected)
    for i, layer in enumerate(layers):
        assert layer["weights"].shape == (layer["output_dim"], layer["input_dim"])
        assert np.abs(layer["weights"]).max() <= 127
        assert layer["input_dim"] <= compile_onnx.MAX_INPUTS
        assert layer["output_dim"] <= compile_onnx.MAX_OUTPUTS
        assert 0 <= layer["output_shift"] <= 31
        assert layer["relu"] == (i < len(layers) - 1)  # ReLU on hidden layers only
    assert np.abs(inputs).max() <= 127
