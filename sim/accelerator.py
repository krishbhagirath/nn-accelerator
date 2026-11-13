# Host-side driver for the accelerator, used by every cocotb test.
#
# This plays the role a real software runtime would. The C++ runtime
# (runtime/main.cpp) does exactly the same steps:
#
#   once per model:      load_network(layers)   layer table + every layer's weights/biases
#   once per inference:  load_inputs(x) -> run(num_layers) -> read_outputs(n)
#
# A single layer is just a 1-layer network (load_layer / run_layer below).
#
# Every host action takes whole clock cycles, and we count them, so the
# benchmark can report load / compute / read-back cycles separately.

import sys
from pathlib import Path

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ReadOnly, RisingEdge
from cocotb.utils import get_sim_time

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))
import golden  # noqa: E402

CLOCK_PERIOD_NS = 10


class Accelerator:

    def __init__(self, dut):
        self.dut = dut
        self.num_macs = int(dut.NUM_MACS.value)
        self.max_inputs = int(dut.MAX_INPUTS.value)
        self.max_outputs = int(dut.MAX_OUTPUTS.value)
        self.max_layers = int(dut.MAX_LAYERS.value)

    def now_cycles(self):
        return int(get_sim_time("ns")) // CLOCK_PERIOD_NS

    async def start(self):
        """Start the clock and reset the accelerator."""
        cocotb.start_soon(Clock(self.dut.clk, CLOCK_PERIOD_NS, unit="ns").start())
        await self.reset()

    async def reset(self, cycles=2):
        dut = self.dut
        dut.reset.value = 1
        dut.start.value = 0
        dut.config_num_layers.value = 1
        dut.layer_write_enable.value = 0
        dut.weight_write_enable.value = 0
        dut.bias_write_enable.value = 0
        dut.input_write_enable.value = 0
        dut.output_read_address.value = 0
        for _ in range(cycles):
            await RisingEdge(dut.clk)
        dut.reset.value = 0
        await RisingEdge(dut.clk)

    # ------------------------------------------------------------
    # Loading a model: one value per clock cycle
    # ------------------------------------------------------------

    async def write_layer_config(self, index, layer):
        """One layer table entry: dims, shift, ReLU."""
        dut = self.dut
        dut.layer_write_enable.value = 1
        dut.layer_write_index.value = index
        dut.layer_write_input_dim.value = layer["input_dim"]
        dut.layer_write_output_dim.value = layer["output_dim"]
        dut.layer_write_output_shift.value = layer["output_shift"]
        dut.layer_write_relu.value = int(bool(layer["relu"]))
        await RisingEdge(dut.clk)
        dut.layer_write_enable.value = 0
        return 1

    async def load_weights(self, index, weights):
        """weights[neuron][input] for layer `index`. Returns the cycles used."""
        dut = self.dut
        cycles = 0
        dut.weight_write_layer.value = index
        for neuron, row in enumerate(weights):
            for i, value in enumerate(row):
                dut.weight_write_enable.value = 1
                dut.weight_write_neuron.value = neuron
                dut.weight_write_input.value = i
                dut.weight_write_data.value = int(value)
                await RisingEdge(dut.clk)
                cycles += 1
        dut.weight_write_enable.value = 0
        return cycles

    async def load_biases(self, index, biases):
        dut = self.dut
        dut.bias_write_layer.value = index
        for neuron, value in enumerate(biases):
            dut.bias_write_enable.value = 1
            dut.bias_write_neuron.value = neuron
            dut.bias_write_data.value = int(value)
            await RisingEdge(dut.clk)
        dut.bias_write_enable.value = 0
        return len(biases)

    async def load_layer(self, layer, index=0):
        """Put one layer (config + weights + biases) into table slot `index`."""
        return (await self.write_layer_config(index, layer)
                + await self.load_weights(index, layer["weights"])
                + await self.load_biases(index, layer["biases"]))

    async def load_network(self, layers):
        """Load every layer of a model. Done once; the weights then stay on chip."""
        assert 1 <= len(layers) <= self.max_layers
        cycles = 0
        for index, layer in enumerate(layers):
            cycles += await self.load_layer(layer, index)
        return cycles

    async def load_inputs(self, inputs):
        dut = self.dut
        for address, value in enumerate(inputs):
            dut.input_write_enable.value = 1
            dut.input_write_address.value = address
            dut.input_write_data.value = int(value)
            await RisingEdge(dut.clk)
        dut.input_write_enable.value = 0
        return len(inputs)

    # ------------------------------------------------------------
    # Running
    # ------------------------------------------------------------

    async def run(self, num_layers):
        """Start the network, wait for done, release start.

        Returns the hardware counters plus host-observed latency:
          start_to_done = clock edges from asserting start until done is seen.
        """
        dut = self.dut
        dut.config_num_layers.value = num_layers
        dut.start.value = 1
        started_at = self.now_cycles()

        await RisingEdge(dut.done)      # simulator jumps straight to done
        start_to_done = self.now_cycles() - started_at

        # done and the counters update on the same clock edge; wait until all
        # of that edge's updates have settled before sampling the counters.
        await ReadOnly()
        counters = {
            "cycle_count": int(dut.cycle_count.value),
            "mac_active_cycles": int(dut.mac_active_cycles.value),
            "mac_operations": int(dut.mac_operations.value),
            "start_to_done": start_to_done,
        }

        # Handshake: drop start so the controller goes DONE -> IDLE
        await RisingEdge(dut.clk)
        dut.start.value = 0
        await RisingEdge(dut.clk)
        return counters

    async def read_outputs(self, count):
        """Read `count` INT8 results, one per clock cycle."""
        dut = self.dut
        results = []
        for address in range(count):
            dut.output_read_address.value = address
            await ReadOnly()
            results.append(dut.output_read_data.value.to_signed())
            await RisingEdge(dut.clk)
        return results

    async def infer(self, layers, inputs):
        """One inference of an already-loaded network: inputs in, final outputs out."""
        await self.load_inputs(inputs)
        counters = await self.run(len(layers))
        outputs = await self.read_outputs(layers[-1]["output_dim"])
        return outputs, counters

    async def run_layer(self, layer, inputs):
        """One already-loaded layer (in slot 0) as a 1-layer network."""
        return await self.infer([layer], inputs)


def expected_layer_counters(input_dim, output_dim, num_macs):
    """What the counters MUST read for one layer (see rtl/controller.sv)."""
    batches = -(-output_dim // num_macs)  # ceil division
    return {
        "cycle_count": batches * input_dim + 2,
        "mac_active_cycles": batches * input_dim,
        "mac_operations": input_dim * output_dim,
        "start_to_done": batches * input_dim + 3,
    }


def expected_network_counters(layers, num_macs):
    """A network is its layers back to back: every counter is a plain sum,
    and start_to_done is 1 more than cycle_count (the cycle that accepts start)."""
    per_layer = [expected_layer_counters(l["input_dim"], l["output_dim"], num_macs) for l in layers]
    total = {key: sum(c[key] for c in per_layer)
             for key in ["cycle_count", "mac_active_cycles", "mac_operations"]}
    total["start_to_done"] = total["cycle_count"] + 1
    return total
