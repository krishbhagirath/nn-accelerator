from pathlib import Path
from cocotb_tools.runner import get_runner

proj = Path(__file__).resolve().parent

sources = [
    proj / "rtl" / "mac_unit.sv",
    proj / "rtl" / "mac_array.sv",
    proj / "rtl" / "input_memory.sv",
    proj / "rtl" / "weight_memory.sv",
    proj / "rtl" / "bias_memory.sv",
    proj / "rtl" / "post_process.sv",
    proj / "rtl" / "output_memory.sv",
    proj / "rtl" / "controller.sv",
    proj / "rtl" / "nn_accelerator_top.sv",
]

runner = get_runner("icarus")

runner.build(
    sources=sources,
    hdl_toplevel="nn_accelerator_top",
    build_args=["-g2012"],
    build_dir=proj / "build" / "cocotb",
)

runner.test(
    hdl_toplevel="nn_accelerator_top",
    test_module="sim.test_nn_accelerator",
    build_dir=proj / "build" / "cocotb",
)