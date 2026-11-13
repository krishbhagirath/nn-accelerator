# Measures the ORIGINAL fixed 8->16, 4-MAC design kept in baseline/rtl/.
#
# It runs the tiny model's first layer (the only thing the old design can run)
# on every tiny test sample, checks it against the golden model, and measures
# latency the same way the new design is measured (accelerator.py):
#   start_to_done = clock edges from asserting start until done is seen
#
# The old RTL hard-codes the shift as a parameter, so the runner builds it with
# OUTPUT_SHIFT = the compiler's shift for that layer.

import json
import os
from pathlib import Path

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ReadOnly, RisingEdge
from cocotb.utils import get_sim_time

from accelerator import golden

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def now_cycles():
    return int(get_sim_time("ns")) // 10


@cocotb.test()
async def test_baseline_design(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    manifest, layers, inputs, _, _ = golden.load_compiled_model(PROJECT_ROOT / "compiled" / "tiny")
    layer = layers[0]
    assert (layer["input_dim"], layer["output_dim"]) == (8, 16)
    assert int(dut.post_process_inst.OUTPUT_SHIFT.value) == layer["output_shift"]

    dut.reset.value = 1
    dut.start.value = 0
    dut.input_write_enable.value = 0
    dut.weight_write_enable.value = 0
    dut.bias_write_enable.value = 0
    dut.output_read_address.value = 0
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)
    dut.reset.value = 0

    # Old design: flat weight address = neuron * 8 + input
    for neuron in range(16):
        for index in range(8):
            dut.weight_write_enable.value = 1
            dut.weight_write_address.value = neuron * 8 + index
            dut.weight_write_data.value = int(layer["weights"][neuron][index])
            await RisingEdge(dut.clk)
    dut.weight_write_enable.value = 0
    for neuron in range(16):
        dut.bias_write_enable.value = 1
        dut.bias_write_address.value = neuron
        dut.bias_write_data.value = int(layer["biases"][neuron])
        await RisingEdge(dut.clk)
    dut.bias_write_enable.value = 0

    expected = golden.fc_layer(inputs, layer["weights"], layer["biases"],
                               layer["output_shift"], True)
    latencies = set()
    busy_cycles = set()
    for sample_number, x in enumerate(inputs):
        for address, value in enumerate(x):
            dut.input_write_enable.value = 1
            dut.input_write_address.value = address
            dut.input_write_data.value = int(value)
            await RisingEdge(dut.clk)
        dut.input_write_enable.value = 0

        dut.start.value = 1
        started_at = now_cycles()
        busy = 0
        while True:
            await RisingEdge(dut.clk)
            await ReadOnly()
            busy += int(dut.busy.value)
            if dut.done.value == 1:
                break
        latencies.add(now_cycles() - started_at)
        busy_cycles.add(busy)

        await RisingEdge(dut.clk)
        dut.start.value = 0
        await RisingEdge(dut.clk)

        outputs = []
        for address in range(16):
            dut.output_read_address.value = address
            await ReadOnly()
            outputs.append(dut.output_read_data.value.to_signed())
            await RisingEdge(dut.clk)
        assert outputs == expected[sample_number].tolist(), f"baseline sample {sample_number} mismatch"

    assert len(latencies) == 1 and len(busy_cycles) == 1
    result = {
        "design": "original fixed 8->16, 4-MAC",
        "layer": "tiny layer 0 (8 -> 16)",
        "samples": len(inputs),
        "bit_exact_samples": len(inputs),
        "start_to_done": latencies.pop(),
        "busy_cycles": busy_cycles.pop(),
        "mac_operations": 8 * 16,
        "num_macs": 4,
    }
    dut._log.info(f"baseline: {result}")
    Path(os.environ["RESULTS_JSON"]).write_text(json.dumps(result, indent=4))
