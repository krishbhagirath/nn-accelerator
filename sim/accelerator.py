# Host-side driver for the accelerator, used by every cocotb test.
#
# This plays the role a real software runtime would: write memories, set the
# layer configuration, pulse start, wait for done, read results back.
# The C++ runtime (runtime/main.cpp) does exactly the same steps.
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
        dut.input_write_enable.value = 0
        dut.weight_write_enable.value = 0
        dut.bias_write_enable.value = 0
        dut.output_read_address.value = 0
        dut.config_input_dim.value = 1
        dut.config_output_dim.value = 1
        dut.config_output_shift.value = 0
        dut.config_relu.value = 0
        for _ in range(cycles):
            await RisingEdge(dut.clk)
        dut.reset.value = 0
        await RisingEdge(dut.clk)

    # ------------------------------------------------------------
    # Loading memories: one value per clock cycle
    # ------------------------------------------------------------

    async def load_weights(self, weights):
        """weights[neuron][input]. Returns the number of cycles used."""
        dut = self.dut
        cycles = 0
        for neuron, row in enumerate(weights):
            for index, value in enumerate(row):
                dut.weight_write_enable.value = 1
                dut.weight_write_neuron.value = neuron
                dut.weight_write_input.value = index
                dut.weight_write_data.value = int(value)
                await RisingEdge(dut.clk)
                cycles += 1
        dut.weight_write_enable.value = 0
        return cycles

    async def load_biases(self, biases):
        dut = self.dut
        for neuron, value in enumerate(biases):
            dut.bias_write_enable.value = 1
            dut.bias_write_neuron.value = neuron
            dut.bias_write_data.value = int(value)
            await RisingEdge(dut.clk)
        dut.bias_write_enable.value = 0
        return len(biases)

    async def load_inputs(self, inputs):
        dut = self.dut
        for address, value in enumerate(inputs):
            dut.input_write_enable.value = 1
            dut.input_write_address.value = address
            dut.input_write_data.value = int(value)
            await RisingEdge(dut.clk)
        dut.input_write_enable.value = 0
        return len(inputs)

    async def load_layer(self, layer):
        """Load one compiled layer's weights and biases."""
        return await self.load_weights(layer["weights"]) + await self.load_biases(layer["biases"])

    # ------------------------------------------------------------
    # Running one layer
    # ------------------------------------------------------------

    async def run(self, input_dim, output_dim, shift, relu):
        """Configure, start, wait for done, release start.

        Returns the hardware counters plus host-observed latency:
          start_to_done = clock edges from asserting start until done is seen.
        """
        dut = self.dut
        dut.config_input_dim.value = input_dim
        dut.config_output_dim.value = output_dim
        dut.config_output_shift.value = shift
        dut.config_relu.value = int(bool(relu))
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

    async def run_layer(self, layer, inputs):
        """Load inputs, run one already-loaded layer, read its outputs."""
        await self.load_inputs(inputs)
        counters = await self.run(layer["input_dim"], layer["output_dim"],
                                  layer["output_shift"], layer["relu"])
        outputs = await self.read_outputs(layer["output_dim"])
        return outputs, counters


def expected_layer_counters(input_dim, output_dim, num_macs):
    """What the counters MUST read for one layer (see rtl/controller.sv)."""
    batches = -(-output_dim // num_macs)  # ceil division
    return {
        "cycle_count": batches * input_dim + 2,
        "mac_active_cycles": batches * input_dim,
        "mac_operations": input_dim * output_dim,
        "start_to_done": batches * input_dim + 3,
    }
