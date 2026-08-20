import random

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge
from pathlib import Path

# Location of the data exported by train_model.py
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DATA_DIR = PROJECT_ROOT / "model_data"


def read_values(filename):
    """Read one integer per line from an exported model-data file."""
    with open(MODEL_DATA_DIR / filename, "r") as f:
        return [
            int(line.strip())
            for line in f
            if line.strip()
        ]

# NUM_TESTS = 100

# def golden_model(inputs, weights, biases):
#     """
#     Python reference for the same 8 -> 16 layer implemented in RTL.
#     """

#     expected = []

#     for neuron in range(16):

#         total = biases[neuron]

#         for i in range(8):
#             total += inputs[i] * weights[neuron][i]

#         # ReLU
#         if total < 0:
#             total = 0

#         # Saturate to positive INT8 range
#         if total > 127:
#             total = 127

#         expected.append(total)

#     return expected


async def reset_dut(dut):
    """
    Return the accelerator to a known state before each test case.
    """

    dut.reset.value = 1
    dut.start.value = 0

    dut.input_write_enable.value = 0
    dut.weight_write_enable.value = 0
    dut.bias_write_enable.value = 0

    dut.output_read_address.value = 0

    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)

    dut.reset.value = 0


async def load_inputs(dut, inputs):

    for address, value in enumerate(inputs):

        dut.input_write_enable.value = 1
        dut.input_write_address.value = address
        dut.input_write_data.value = value

        await RisingEdge(dut.clk)

    dut.input_write_enable.value = 0


async def load_weights(dut, weights):

    address = 0

    for neuron_weights in weights:
        for value in neuron_weights:

            dut.weight_write_enable.value = 1
            dut.weight_write_address.value = address
            dut.weight_write_data.value = value

            await RisingEdge(dut.clk)

            address += 1

    dut.weight_write_enable.value = 0


async def load_biases(dut, biases):

    for address, value in enumerate(biases):

        dut.bias_write_enable.value = 1
        dut.bias_write_address.value = address
        dut.bias_write_data.value = value

        await RisingEdge(dut.clk)

    dut.bias_write_enable.value = 0


async def run_inference(dut):

    dut.start.value = 1

    while dut.done.value != 1:
        await RisingEdge(dut.clk)


async def read_outputs(dut):

    outputs = []

    for address in range(16):

        dut.output_read_address.value = address

        # Give combinational output time to settle
        await RisingEdge(dut.clk)

        outputs.append(
            dut.output_read_data.value.to_signed()
        )

    return outputs

@cocotb.test()
async def test_trained_model(dut):

    # Start accelerator clock
    cocotb.start_soon(
        Clock(dut.clk, 10, unit="ns").start()
    )

    # Put accelerator into a known starting state
    await reset_dut(dut)


    # ---------------------------------------------------------
    # Load the trained/quantized model exported by train_model.py
    # ---------------------------------------------------------

    inputs = read_values("sample_input.txt")
    flat_weights = read_values("weights.txt")
    biases = read_values("biases.txt")
    expected = read_values("expected_output.txt")


    # weights.txt contains 128 sequential values.
    # Convert them back into 16 neurons x 8 weights.
    weights = [
        flat_weights[i * 8:(i + 1) * 8]
        for i in range(16)
    ]


    # ---------------------------------------------------------
    # Load trained model into the simulated accelerator
    # ---------------------------------------------------------

    await load_inputs(dut, inputs)
    await load_weights(dut, weights)
    await load_biases(dut, biases)


    # ---------------------------------------------------------
    # Run hardware inference
    # ---------------------------------------------------------

    await run_inference(dut)

    rtl_outputs = await read_outputs(dut)


    # ---------------------------------------------------------
    # Verify RTL matches quantized Python result
    # ---------------------------------------------------------

    assert rtl_outputs == expected, (
        f"\nTrained model inference FAILED"
        f"\nInput:    {inputs}"
        f"\nExpected: {expected}"
        f"\nRTL:      {rtl_outputs}"
    )

    dut._log.info(f"Expected: {expected}")
    dut._log.info(f"RTL:      {rtl_outputs}")
    dut._log.info("TRAINED MODEL RTL INFERENCE PASSED")

    # Release START so controller can return to IDLE
    dut.start.value = 0

    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)

# @cocotb.test()
# async def test_randomized_inference(dut):

#     # Start one clock for the entire test suite
#     cocotb.start_soon(
#         Clock(dut.clk, 10, unit="ns").start()
#     )

#     # Fixed seed makes failures reproducible
#     random.seed(42)

#     for test_number in range(NUM_TESTS):

#         await reset_dut(dut)

#         # Keep values modest initially so we test logic
#         # without making overflow analysis complicated yet.
#         inputs = [
#             random.randint(-8, 8)
#             for _ in range(8)
#         ]

#         weights = [
#             [
#                 random.randint(-4, 4)
#                 for _ in range(8)
#             ]
#             for _ in range(16)
#         ]

#         biases = [
#             random.randint(-16, 16)
#             for _ in range(16)
#         ]

#         expected = golden_model(
#             inputs,
#             weights,
#             biases
#         )

#         await load_inputs(dut, inputs)
#         await load_weights(dut, weights)
#         await load_biases(dut, biases)

#         await run_inference(dut)

#         rtl_outputs = await read_outputs(dut)

#         assert rtl_outputs == expected, (
#             f"\nTest {test_number} FAILED"
#             f"\nInputs:   {inputs}"
#             f"\nExpected: {expected}"
#             f"\nRTL:      {rtl_outputs}"
#         )

#         dut._log.info(
#             f"Random test {test_number + 1}/{NUM_TESTS}: PASS"
#         )

#         # Release START so controller returns from DONE -> IDLE
#         dut.start.value = 0

#         await RisingEdge(dut.clk)
#         await RisingEdge(dut.clk)

#     dut._log.info(
#         f"ALL {NUM_TESTS} RANDOMIZED TESTS PASSED"
#     )