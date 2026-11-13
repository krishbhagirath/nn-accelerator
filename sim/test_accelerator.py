# Self-checking cocotb regression for the accelerator.
#
# Every test compares the RTL against python/golden.py (the canonical integer
# model) and asserts. The same tests run for every NUM_MACS build.
#
# Single-layer tests run each layer as a 1-layer network; the network tests
# chain several layers on chip.
#
# Run through:  python run_pipeline.py test

import random
from pathlib import Path

import cocotb
import numpy as np
from cocotb.triggers import RisingEdge

from accelerator import Accelerator, expected_layer_counters, expected_network_counters, golden

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def random_layer(rng, input_dim, output_dim, shift=None, relu=None,
                 weight_range=(-128, 127), bias_range=(-50000, 50000)):
    return {
        "input_dim": input_dim,
        "output_dim": output_dim,
        "weights": np.array([[rng.randint(*weight_range) for _ in range(input_dim)]
                             for _ in range(output_dim)], dtype=np.int64),
        "biases": np.array([rng.randint(*bias_range) for _ in range(output_dim)], dtype=np.int64),
        "output_shift": rng.randint(0, 14) if shift is None else shift,
        "relu": rng.random() < 0.5 if relu is None else relu,
    }


async def check_layer(accel, layer, inputs, name):
    """Load + run one layer, assert outputs and counters are exactly right."""
    await accel.load_layer(layer)
    outputs, counters = await accel.run_layer(layer, inputs)

    expected = golden.fc_layer(np.array(inputs, dtype=np.int64), layer["weights"],
                               layer["biases"], layer["output_shift"], layer["relu"])
    assert outputs == expected.tolist(), (
        f"{name}: output mismatch\n  RTL:      {outputs}\n  expected: {expected.tolist()}")

    want = expected_layer_counters(layer["input_dim"], layer["output_dim"], accel.num_macs)
    assert counters == want, f"{name}: counters {counters} != expected {want}"
    return outputs


@cocotb.test()
async def test_reset_state(dut):
    """After reset the engine is idle and the counters are zero."""
    accel = Accelerator(dut)
    await accel.start()
    assert dut.busy.value == 0 and dut.done.value == 0
    assert int(dut.cycle_count.value) == 0
    assert int(dut.mac_active_cycles.value) == 0
    assert int(dut.mac_operations.value) == 0


@cocotb.test()
async def test_start_done_handshake(dut):
    """busy during compute; done held while start is high; clean restart."""
    accel = Accelerator(dut)
    await accel.start()
    rng = random.Random(1)
    layer = random_layer(rng, 8, 16, shift=6, relu=True)
    inputs = [rng.randint(-128, 127) for _ in range(8)]
    await accel.load_layer(layer)
    await accel.load_inputs(inputs)

    dut.config_num_layers.value = 1
    dut.start.value = 1
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)
    assert dut.busy.value == 1 and dut.done.value == 0

    # Changing the configuration mid-run must not matter: the layer count was
    # captured at start, and the layer's table entry when the layer began.
    dut.config_num_layers.value = 3
    await accel.write_layer_config(0, {"input_dim": 3, "output_dim": 1, "output_shift": 0, "relu": False})

    await RisingEdge(dut.done)
    for _ in range(5):  # done stays high as long as start stays high
        await RisingEdge(dut.clk)
        assert dut.done.value == 1 and dut.busy.value == 0

    dut.start.value = 0
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)
    assert dut.done.value == 0 and dut.busy.value == 0

    first = await accel.read_outputs(16)
    expected = golden.fc_layer(np.array(inputs), layer["weights"], layer["biases"], 6, True)
    assert first == expected.tolist()

    # Second run with no reset in between gives the same answer (restore the
    # table entry we overwrote, and reload the inputs: the output buffer is now
    # the current one)
    await accel.write_layer_config(0, layer)
    await accel.load_inputs(inputs)
    counters = await accel.run(1)
    assert await accel.read_outputs(16) == first
    assert counters == expected_layer_counters(8, 16, accel.num_macs)


@cocotb.test()
async def test_reset_during_inference(dut):
    """Reset in the middle of a run returns to idle, and the next run is correct."""
    accel = Accelerator(dut)
    await accel.start()
    rng = random.Random(2)
    layer = random_layer(rng, 20, 12)
    inputs = [rng.randint(-128, 127) for _ in range(20)]
    await accel.load_layer(layer)
    await accel.load_inputs(inputs)

    dut.config_num_layers.value = 1
    dut.start.value = 1
    for _ in range(7):
        await RisingEdge(dut.clk)
    assert dut.busy.value == 1

    dut.reset.value = 1
    dut.start.value = 0
    await RisingEdge(dut.clk)
    dut.reset.value = 0
    await RisingEdge(dut.clk)
    assert dut.busy.value == 0 and dut.done.value == 0
    assert int(dut.cycle_count.value) == 0

    # Weights and the layer table survive reset. Reset points the host back at
    # buffer 0, which still holds the inputs, so just run again.
    counters = await accel.run(1)
    outputs = await accel.read_outputs(12)
    expected = golden.fc_layer(np.array(inputs), layer["weights"], layer["biases"],
                               layer["output_shift"], layer["relu"])
    assert outputs == expected.tolist()
    assert counters == expected_layer_counters(20, 12, accel.num_macs)


@cocotb.test()
async def test_random_layers(dut):
    """Random sizes (incl. 1x1, max size, non-multiples of NUM_MACS), full-range values."""
    accel = Accelerator(dut)
    await accel.start()
    rng = random.Random(1234)

    sizes = [(1, 1), (1, 32), (64, 1), (64, 32), (3, 5), (7, 9), (8, 16), (30, 16),
             (4, 8), (16, 2), (64, 10), (33, 31)]
    sizes += [(rng.randint(1, 64), rng.randint(1, 32)) for _ in range(28)]

    for case, (input_dim, output_dim) in enumerate(sizes):
        layer = random_layer(rng, input_dim, output_dim)
        inputs = [rng.randint(-128, 127) for _ in range(input_dim)]
        await check_layer(accel, layer, inputs, f"random case {case} ({input_dim}x{output_dim})")
    dut._log.info(f"{len(sizes)} random layers bit-exact with NUM_MACS={accel.num_macs}")


@cocotb.test()
async def test_extreme_values(dut):
    """INT8 extremes, zeros, saturation in both directions, large +/- biases."""
    accel = Accelerator(dut)
    await accel.start()

    def layer(input_dim, output_dim, weight, bias, shift, relu):
        return {
            "input_dim": input_dim, "output_dim": output_dim,
            "weights": np.full((output_dim, input_dim), weight, dtype=np.int64),
            "biases": np.full(output_dim, bias, dtype=np.int64),
            "output_shift": shift, "relu": relu,
        }

    cases = [
        # name, layer, inputs
        ("max positive accumulator, saturates to 127",
         layer(64, 32, -128, 0, 0, False), [-128] * 64),
        ("max negative accumulator, saturates to -128",
         layer(64, 32, -128, 0, 0, False), [127] * 64),
        ("max negative accumulator with ReLU -> 0",
         layer(64, 32, -128, 0, 0, True), [127] * 64),
        ("largest accumulator shifted down to fit",
         layer(64, 32, -128, 0, 13, False), [-128] * 64),
        ("zero inputs: output is just bias >> shift",
         layer(10, 7, 99, 1000, 3, False), [0] * 10),
        ("zero weights: output is just bias >> shift",
         layer(10, 7, 0, -1000, 3, False), [127] * 10),
        ("huge positive bias saturates",
         layer(5, 5, 1, 2**31 - 2**20, 0, True), [1] * 5),
        ("huge negative bias saturates",
         layer(5, 5, 1, -(2**31) + 2**20, 0, False), [1] * 5),
        ("negative value shift rounds toward -infinity",
         layer(1, 1, 1, -1, 1, False), [0]),
        ("maximum shift of 31",
         layer(64, 3, 127, 2**30, 31, False), [127] * 64),
    ]
    for name, lay, inputs in cases:
        outputs = await check_layer(accel, lay, inputs, name)
        dut._log.info(f"PASS {name}: {outputs[:4]}")

    # Exact clamp boundaries: (127 << s) stays 127, (128 << s) clamps, -1 -> -1 or 0
    shift = 4
    biases = np.array([127 << shift, (128 << shift) - 1, 128 << shift, -1,
                       -(128 << shift), -(129 << shift)], dtype=np.int64)
    boundary = {"input_dim": 1, "output_dim": 6, "weights": np.zeros((6, 1), dtype=np.int64),
                "biases": biases, "output_shift": shift, "relu": False}
    assert await check_layer(accel, boundary, [0], "signed boundaries") == [127, 127, 127, -1, -128, -128]
    boundary["relu"] = True
    assert await check_layer(accel, boundary, [0], "relu boundaries") == [127, 127, 127, 0, 0, 0]


@cocotb.test()
async def test_cycle_counts(dut):
    """Counters follow the documented formula for many layer shapes."""
    accel = Accelerator(dut)
    await accel.start()
    rng = random.Random(5)
    for input_dim, output_dim in [(1, 1), (8, 16), (8, 15), (8, 17), (64, 32), (30, 2), (5, 31)]:
        lay = random_layer(rng, input_dim, output_dim)
        await accel.load_layer(lay)
        await accel.load_inputs([0] * input_dim)
        counters = await accel.run(1)
        want = expected_layer_counters(input_dim, output_dim, accel.num_macs)
        assert counters == want, f"{input_dim}x{output_dim}: {counters} != {want}"
        # Sanity: every useful MAC is counted exactly once
        assert counters["mac_operations"] == input_dim * output_dim


def random_network(rng, num_layers):
    """Random chained layers: layer i's output_dim is layer i+1's input_dim."""
    dims = [rng.randint(1, 64)] + [rng.randint(1, 32) for _ in range(num_layers)]
    layers = [random_layer(rng, dims[i], dims[i + 1],
                           weight_range=(-40, 40), bias_range=(-3000, 3000))
              for i in range(num_layers)]
    # Hidden layers usually use ReLU, but test both
    for layer in layers:
        layer["output_shift"] = rng.randint(4, 10)
    return layers


async def check_network(accel, layers, samples, name):
    """Load a network once, run every sample through it, check every output and counter."""
    await accel.load_network(layers)
    want = expected_network_counters(layers, accel.num_macs)
    for number, x in enumerate(samples):
        outputs, counters = await accel.infer(layers, x)
        expected = golden.run_model(np.array([x], dtype=np.int64), layers)[-1][0]
        assert outputs == expected.tolist(), (
            f"{name} sample {number}: output mismatch\n  RTL:      {outputs}\n"
            f"  expected: {expected.tolist()}")
        assert counters == want, f"{name} sample {number}: counters {counters} != {want}"


@cocotb.test()
async def test_random_networks(dut):
    """Random 1-4 layer networks, several back-to-back inferences each.

    Back-to-back runs with odd and even layer counts check that the ping-pong
    buffers always hand the host the right buffer.
    """
    accel = Accelerator(dut)
    await accel.start()
    rng = random.Random(77)
    count = 0
    for num_layers in [1, 2, 3, 4] * 6:
        layers = random_network(rng, num_layers)
        samples = [[rng.randint(-128, 127) for _ in range(layers[0]["input_dim"])]
                   for _ in range(3)]
        dims = " -> ".join(str(d) for d in [layers[0]["input_dim"]] + [l["output_dim"] for l in layers])
        await check_network(accel, layers, samples, f"network {count} ({dims})")
        count += 1
    dut._log.info(f"{count} random networks bit-exact with NUM_MACS={accel.num_macs}")


@cocotb.test()
async def test_max_size_network(dut):
    """A full MAX_LAYERS-deep network at the largest layer sizes."""
    accel = Accelerator(dut)
    await accel.start()
    rng = random.Random(9)
    dims = [64, 32, 32, 32, 32]
    layers = [random_layer(rng, dims[i], dims[i + 1], shift=rng.randint(6, 12),
                           relu=i < 3, bias_range=(-20000, 20000)) for i in range(4)]
    samples = [[rng.randint(-128, 127) for _ in range(64)] for _ in range(3)]
    await check_network(accel, layers, samples, "64->32->32->32->32")


@cocotb.test()
async def test_compiled_models(dut):
    """First 20 test samples of every compiled model, whole network on chip, bit-exact."""
    accel = Accelerator(dut)
    await accel.start()

    compiled_root = PROJECT_ROOT / "compiled"
    model_dirs = sorted(p for p in compiled_root.iterdir() if (p / "manifest.json").exists())
    assert model_dirs, "no compiled models found - run: python run_pipeline.py compile"

    for model_dir in model_dirs:
        manifest, layers, inputs, expected, _ = golden.load_compiled_model(model_dir)
        await check_network(accel, layers, inputs[:20].tolist(), manifest["model"])
        dut._log.info(f"{manifest['model']}: 20 samples bit-exact through {len(layers)} layers on chip")
