# Runs a whole compiled model on the RTL and saves the results as JSON.
#
# Environment variables (set by run_pipeline.py):
#   MODEL         name of a folder in compiled/
#   MAX_SAMPLES   optional limit on the number of test samples
#   RESULTS_JSON  where to write the results
#
# Execution is weight-stationary, like a real accelerator running a batch:
#   for each layer:
#       load this layer's weights + biases once
#       for each sample: load its inputs, start, wait for done, read outputs
# The outputs of layer i become the inputs of layer i+1.
#
# Cycle accounting (all measured, in clock cycles):
#   weight_load_cycles   one-time cost per layer (weights + biases)
#   cycle_count          hardware counter: compute cycles of one layer run
#   start_to_done        host view of one layer run (= cycle_count + 1)
#   host_cycles          one sample through one layer, everything the host
#                        does: write inputs + start..done + handshake + read outputs

import json
import os
from pathlib import Path

import cocotb

from accelerator import Accelerator, golden

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@cocotb.test()
async def test_full_model(dut):
    accel = Accelerator(dut)
    await accel.start()

    model = os.environ["MODEL"]
    manifest, layers, inputs, expected, labels = golden.load_compiled_model(
        PROJECT_ROOT / "compiled" / model)
    limit = int(os.environ.get("MAX_SAMPLES", "0")) or len(inputs)
    inputs, expected, labels = inputs[:limit], expected[:limit], labels[:limit]
    golden_per_layer = golden.run_model(inputs, layers)

    layer_reports = []
    activations = inputs.tolist()
    for index, layer in enumerate(layers):
        weight_load_cycles = await accel.load_layer(layer)

        outputs_this_layer = []
        counters_seen = set()
        host_cycles_seen = set()
        for x in activations:
            before = accel.now_cycles()
            outputs, counters = await accel.run_layer(layer, x)
            host_cycles_seen.add(accel.now_cycles() - before)
            counters_seen.add(tuple(sorted(counters.items())))
            outputs_this_layer.append(outputs)

        # The engine is deterministic: every sample takes the same cycles.
        assert len(counters_seen) == 1 and len(host_cycles_seen) == 1
        counters = dict(counters_seen.pop())

        mismatches = sum(
            1 for rtl, gold in zip(outputs_this_layer, golden_per_layer[index].tolist())
            if rtl != gold)
        layer_reports.append({
            "layer": index,
            "input_dim": layer["input_dim"],
            "output_dim": layer["output_dim"],
            "relu": layer["relu"],
            "output_shift": layer["output_shift"],
            "weight_load_cycles": weight_load_cycles,
            "host_cycles": host_cycles_seen.pop(),
            "sample_mismatches": mismatches,
            **counters,
        })
        activations = outputs_this_layer

    rtl_outputs = activations
    result = {
        "model": model,
        "num_macs": accel.num_macs,
        "samples": len(inputs),
        "layers": layer_reports,
        "rtl_outputs": rtl_outputs,
        "rtl_predictions": golden.predict(rtl_outputs).tolist(),
    }
    Path(os.environ["RESULTS_JSON"]).write_text(json.dumps(result))

    exact = sum(1 for rtl, gold in zip(rtl_outputs, expected.tolist()) if rtl == gold)
    dut._log.info(f"{model} NUM_MACS={accel.num_macs}: {exact}/{len(inputs)} samples bit-exact")
    assert exact == len(inputs), f"{len(inputs) - exact} samples differ from the golden model"
