# Runs a whole compiled model on the RTL and saves the results as JSON.
#
# Environment variables (set by run_pipeline.py):
#   MODEL         name of a folder in compiled/
#   MAX_SAMPLES   optional limit on the number of test samples
#   RESULTS_JSON  where to write the results
#
# Execution is the way a deployed accelerator would serve requests, one
# inference at a time:
#   once:            load the layer table + every layer's weights and biases
#   for each sample: write inputs -> start -> wait done -> read final outputs
# All layers run on chip; the host never sees intermediate activations.
#
# Cycle accounting (all measured, in clock cycles):
#   weight_load_cycles   one-time cost: layer table + all weights + biases
#   cycle_count          hardware counter: compute cycles of one whole inference
#   start_to_done        host view of one run (= cycle_count + 1)
#   host_cycles          one inference, everything the host does:
#                        write inputs + start..done + handshake + read outputs
#
# Two extra measurements, for comparisons in results/:
#   layer0               layer 0 alone as a 1-layer network (same workload as
#                        the original fixed-function design in baseline/rtl)
#   layer_at_a_time      the same inference done the way the previous engine
#                        had to: weights for only ONE layer on chip, so the host
#                        reloads each layer and copies activations between layers

import json
import os
from pathlib import Path

import cocotb

from accelerator import Accelerator, golden

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LAYER_AT_A_TIME_SAMPLES = 3   # deterministic, so a few samples are enough


@cocotb.test()
async def test_full_model(dut):
    accel = Accelerator(dut)
    await accel.start()

    model = os.environ["MODEL"]
    manifest, layers, inputs, expected, labels = golden.load_compiled_model(
        PROJECT_ROOT / "compiled" / model)
    limit = int(os.environ.get("MAX_SAMPLES", "0")) or len(inputs)
    inputs, expected = inputs[:limit], expected[:limit]
    num_layers = len(layers)

    # ---------------- whole network on chip ----------------
    weight_load_cycles = await accel.load_network(layers)

    rtl_outputs = []
    counters_seen = set()
    host_cycles_seen = set()
    for x in inputs.tolist():
        before = accel.now_cycles()
        outputs, counters = await accel.infer(layers, x)
        host_cycles_seen.add(accel.now_cycles() - before)
        counters_seen.add(tuple(sorted(counters.items())))
        rtl_outputs.append(outputs)

    # The engine is deterministic: every sample takes the same cycles.
    assert len(counters_seen) == 1 and len(host_cycles_seen) == 1
    counters = dict(counters_seen.pop())
    host_cycles = host_cycles_seen.pop()

    # ---------------- layer 0 alone ----------------
    # Slot 0 still holds layer 0, so just run a 1-layer network on sample 0.
    await accel.load_inputs(inputs[0].tolist())
    layer0 = await accel.run(1)
    layer0_out = await accel.read_outputs(layers[0]["output_dim"])
    assert layer0_out == golden.run_model(inputs[:1], layers)[0][0].tolist()

    # ---------------- layer at a time (previous engine's flow) ----------------
    layer_at_a_time = set()
    for x, want in zip(inputs[:LAYER_AT_A_TIME_SAMPLES].tolist(),
                       expected[:LAYER_AT_A_TIME_SAMPLES].tolist()):
        before = accel.now_cycles()
        activations = x
        for layer in layers:
            await accel.load_layer(layer, 0)
            activations, _ = await accel.run_layer(layer, activations)
        layer_at_a_time.add(accel.now_cycles() - before)
        assert activations == want
    assert len(layer_at_a_time) == 1

    result = {
        "model": model,
        "num_macs": accel.num_macs,
        "samples": len(inputs),
        "num_layers": num_layers,
        "layers": [{k: layer[k] for k in ["input_dim", "output_dim", "relu", "output_shift"]}
                   for layer in layers],
        "weight_load_cycles": weight_load_cycles,
        "host_cycles": host_cycles,
        "layer_at_a_time_host_cycles": layer_at_a_time.pop(),
        "layer0": layer0,
        **counters,
        "rtl_outputs": rtl_outputs,
        "rtl_predictions": golden.predict(rtl_outputs).tolist(),
    }
    Path(os.environ["RESULTS_JSON"]).write_text(json.dumps(result))

    exact = sum(1 for rtl, gold in zip(rtl_outputs, expected.tolist()) if rtl == gold)
    dut._log.info(f"{model} NUM_MACS={accel.num_macs}: {exact}/{len(inputs)} samples bit-exact, "
                  f"{host_cycles} host cycles/inference "
                  f"(layer-at-a-time: {result['layer_at_a_time_host_cycles']})")
    assert exact == len(inputs), f"{len(inputs) - exact} samples differ from the golden model"
