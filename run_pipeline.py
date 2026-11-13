# One entry point for the whole project.
#
#   python run_pipeline.py train        train every model, export ONNX
#   python run_pipeline.py compile      ONNX -> INT8 files in compiled/
#   python run_pipeline.py lint         Verilator lint for NUM_MACS = 1, 2, 4, 8
#   python run_pipeline.py test         Python unit tests + cocotb regression for 1/2/4/8 MACs
#   python run_pipeline.py benchmark    every model x every MAC count + original design -> results/
#   python run_pipeline.py synth        Yosys + nextpnr for 1/2/4/8 MACs -> results/
#   python run_pipeline.py report       rebuild results/summary.md from the CSV files
#   python run_pipeline.py runtime      build and run the C++ Verilator runtime
#   python run_pipeline.py demo --model digits --num-macs 4
#   python run_pipeline.py all          everything above, in order

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "python"))
import golden  # noqa: E402

RTL_DIR = PROJECT_ROOT / "rtl"
SIM_DIR = PROJECT_ROOT / "sim"
BUILD_DIR = PROJECT_ROOT / "build"
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS = ["tiny", "iris", "cancer", "digits", "digits_deep"]
MAC_COUNTS = [1, 2, 4, 8]

# On Windows the EDA tools (and a modern g++) come from MSYS2. Put them FIRST
# on PATH so an older compiler elsewhere on PATH is not picked up instead.
for tool_dir in [r"C:\msys64\usr\bin", r"C:\msys64\ucrt64\bin"]:
    if Path(tool_dir).exists():
        os.environ["PATH"] = tool_dir + os.pathsep + os.environ["PATH"]

# torch.onnx prints emoji; make sure the console can print them.
os.environ.setdefault("PYTHONIOENCODING", "utf-8")


def verilator():
    """On Windows `verilator` is a Perl script that subprocess can't launch,
    so call the real binary directly (that is all the script does)."""
    if os.name == "nt" and shutil.which("verilator_bin"):
        root = Path(shutil.which("verilator_bin")).parent.parent / "share" / "verilator"
        # Forward slashes: the generated makefile runs commands through sh.
        os.environ.setdefault("VERILATOR_ROOT", root.as_posix())
        return "verilator_bin"
    return "verilator"


def make():
    return "mingw32-make" if os.name == "nt" and shutil.which("mingw32-make") else "make"


def run(command, **kwargs):
    print("$", " ".join(str(c) for c in command))
    subprocess.run([str(c) for c in command], check=True, **kwargs)


def rtl_sources(rtl_dir=RTL_DIR):
    return sorted(rtl_dir.glob("*.sv"))


# ------------------------------------------------------------
# Simulation helpers (cocotb + Icarus Verilog)
# ------------------------------------------------------------

def simulate(test_module, num_macs, env=None, rtl_dir=RTL_DIR, parameters=None, build_name=None):
    """Build the RTL with the given parameters and run one cocotb test module."""
    from cocotb_tools.runner import get_runner

    parameters = parameters if parameters is not None else {"NUM_MACS": num_macs}
    build_dir = BUILD_DIR / (build_name or f"sim_macs{num_macs}")
    runner = get_runner("icarus")
    runner.build(
        sources=rtl_sources(rtl_dir),
        hdl_toplevel="nn_accelerator_top",
        build_args=["-g2012"],
        parameters=parameters,
        build_dir=build_dir,
        always=True,
    )
    results_xml = runner.test(
        hdl_toplevel="nn_accelerator_top",
        test_module=test_module,
        test_dir=SIM_DIR,
        build_dir=build_dir,
        extra_env=env or {},
    )
    # Fail loudly if any cocotb test failed
    from cocotb_tools.runner import get_results
    total, failed = get_results(results_xml)
    if failed:
        raise SystemExit(f"{failed}/{total} cocotb tests FAILED ({test_module}, NUM_MACS={num_macs})")
    return total


def simulate_model(model, num_macs, max_samples=0):
    """Run one compiled model on the RTL; returns the JSON written by sim/test_model.py."""
    out = BUILD_DIR / "results" / f"{model}_macs{num_macs}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    simulate("test_model", num_macs, env={
        "MODEL": model, "RESULTS_JSON": str(out), "MAX_SAMPLES": str(max_samples)})
    return json.loads(out.read_text())


# ------------------------------------------------------------
# Accuracy helpers
# ------------------------------------------------------------

def float_predictions(model):
    """Run the float ONNX model with onnxruntime (independent of our code)."""
    import onnxruntime
    session = onnxruntime.InferenceSession(str(PROJECT_ROOT / "models" / model / "model.onnx"))
    test_inputs = np.loadtxt(PROJECT_ROOT / "models" / model / "test_inputs.txt",
                             ndmin=2, dtype=np.float32)
    # The exported graph has batch size 1, so feed one sample at a time.
    logits = [session.run(None, {"input": x[None, :]})[0][0] for x in test_inputs]
    return np.argmax(np.array(logits), axis=1)


def evaluate_model(model, rtl_result):
    """Combine float, quantized-Python and RTL results into one summary row."""
    manifest, layers, inputs, expected, labels = golden.load_compiled_model(
        PROJECT_ROOT / "compiled" / model)
    n = rtl_result["samples"]
    info = json.loads((PROJECT_ROOT / "models" / model / "info.json").read_text())

    float_pred = float_predictions(model)[:n]
    quant_pred = golden.predict(expected[:n])
    rtl_outputs = np.array(rtl_result["rtl_outputs"])
    rtl_pred = np.array(rtl_result["rtl_predictions"])
    labels = labels[:n]

    compute_cycles = rtl_result["cycle_count"]
    mac_ops = rtl_result["mac_operations"]
    num_macs = rtl_result["num_macs"]

    return {
        "model": model,
        "dataset": info["dataset"],
        "architecture": info["architecture"],
        "parameters": info["num_parameters"],
        "num_macs": num_macs,
        "samples": n,
        "float_accuracy": float((float_pred == labels).mean()),
        "quantized_accuracy": float((quant_pred == labels).mean()),
        "rtl_accuracy": float((rtl_pred == labels).mean()),
        "rtl_vs_quantized_prediction_agreement": float((rtl_pred == quant_pred).mean()),
        "bit_exact_rate": float(np.all(rtl_outputs == expected[:n], axis=1).mean()),
        "integer_mismatches": int((rtl_outputs != expected[:n]).sum()),
        "compute_cycles": compute_cycles,
        "mac_active_cycles": rtl_result["mac_active_cycles"],
        "mac_operations": mac_ops,
        "mac_utilization": mac_ops / (num_macs * compute_cycles),
        "macs_per_cycle": mac_ops / compute_cycles,
        "host_cycles_per_sample": rtl_result["host_cycles"],
        "layer_at_a_time_host_cycles": rtl_result["layer_at_a_time_host_cycles"],
        "host_speedup": rtl_result["layer_at_a_time_host_cycles"] / rtl_result["host_cycles"],
        "weight_load_cycles": rtl_result["weight_load_cycles"],
        "float_predictions": float_pred.tolist(),
        "quantized_predictions": quant_pred.tolist(),
        "labels": labels.tolist(),
    }


def write_csv(path, rows, columns):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in row.items()})


# ------------------------------------------------------------
# Commands
# ------------------------------------------------------------

def cmd_train(args):
    run([sys.executable, PROJECT_ROOT / "python" / "train_model.py", "--model", "all"])


def cmd_compile(args):
    run([sys.executable, PROJECT_ROOT / "python" / "compile_onnx.py", "--model", "all"])


def cmd_lint(args):
    for n in MAC_COUNTS:
        run([verilator(), "--lint-only", "-Wall", f"-GNUM_MACS={n}",
             *rtl_sources(), "--top-module", "nn_accelerator_top"])
        print(f"lint clean: NUM_MACS={n}")


def cmd_test(args):
    run([sys.executable, "-m", "pytest", "-q", PROJECT_ROOT / "tests"])
    for n in MAC_COUNTS:
        total = simulate("test_accelerator", n)
        print(f"cocotb: {total} tests passed with NUM_MACS={n}")


def cmd_benchmark(args):
    RESULTS_DIR.mkdir(exist_ok=True)

    # 1. The original design, for the before/after comparison
    tiny_shift = json.loads((PROJECT_ROOT / "compiled" / "tiny" / "manifest.json").read_text())
    tiny_shift = tiny_shift["layers"][0]["output_shift"]
    baseline_json = BUILD_DIR / "results" / "baseline.json"
    baseline_json.parent.mkdir(parents=True, exist_ok=True)
    simulate("test_baseline", 4, env={"RESULTS_JSON": str(baseline_json)},
             rtl_dir=PROJECT_ROOT / "baseline" / "rtl",
             parameters={"OUTPUT_SHIFT": tiny_shift}, build_name="sim_baseline")
    baseline = json.loads(baseline_json.read_text())

    # 2. Every model on every MAC count
    rows = []
    for model in MODELS:
        for n in MAC_COUNTS:
            rows.append(evaluate_model(model, simulate_model(model, n, args.max_samples)))

    # Speedup relative to the 1-MAC build of the same model
    one_mac = {r["model"]: r["compute_cycles"] for r in rows if r["num_macs"] == 1}
    for r in rows:
        r["speedup_vs_1mac"] = one_mac[r["model"]] / r["compute_cycles"]

    # The new design on the exact workload the old design ran: tiny layer 0
    new_tiny = json.loads((BUILD_DIR / "results" / "tiny_macs4.json").read_text())["layer0"]

    write_csv(RESULTS_DIR / "model_results.csv",
              [r for r in rows if r["num_macs"] == 4],
              ["model", "dataset", "architecture", "parameters", "samples", "float_accuracy",
               "quantized_accuracy", "rtl_accuracy", "bit_exact_rate", "integer_mismatches",
               "compute_cycles", "host_cycles_per_sample", "layer_at_a_time_host_cycles",
               "host_speedup", "weight_load_cycles"])
    write_csv(RESULTS_DIR / "mac_scaling.csv", rows,
              ["model", "num_macs", "compute_cycles", "speedup_vs_1mac", "mac_active_cycles",
                  "mac_operations", "mac_utilization", "macs_per_cycle", "host_cycles_per_sample",
               "layer_at_a_time_host_cycles", "host_speedup", "rtl_accuracy", "integer_mismatches"])
    write_csv(RESULTS_DIR / "baseline_comparison.csv", [
        {"design": "original (baseline/rtl)", "workload": "tiny layer 0, 8->16",
         "samples": baseline["samples"], "start_to_done": baseline["start_to_done"],
         "mac_operations": baseline["mac_operations"],
         "mac_utilization": baseline["mac_operations"] / (4 * baseline["start_to_done"])},
        {"design": "new generic engine, NUM_MACS=4", "workload": "tiny layer 0, 8->16",
         "samples": 1,
         "start_to_done": new_tiny["start_to_done"], "mac_operations": new_tiny["mac_operations"],
         "mac_utilization": new_tiny["mac_operations"] / (4 * new_tiny["start_to_done"])},
    ], ["design", "workload", "samples", "start_to_done", "mac_operations", "mac_utilization"])

    (RESULTS_DIR / "predictions.json").write_text(json.dumps(
        {f"{r['model']}_macs{r['num_macs']}": {k: r[k] for k in
         ["labels", "float_predictions", "quantized_predictions"]} for r in rows if r["num_macs"] == 4}))
    cmd_report(args)


def read_results(name):
    path = RESULTS_DIR / name
    if not path.exists():
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def cmd_report(args):
    """Build results/summary.md (and throughput.csv) from the generated CSVs only."""
    models = read_results("model_results.csv")
    scaling = read_results("mac_scaling.csv")
    baseline = read_results("baseline_comparison.csv")
    synth = {r["num_macs"]: r for r in read_results("synthesis_results.csv")}
    pct = lambda v: f"{float(v) * 100:.2f}%"

    lines = ["# Benchmark summary", "",
             "Generated by `python run_pipeline.py report` from the CSV files in this folder.", "",
             "- Cycle numbers are MEASURED in RTL simulation (Icarus Verilog + cocotb).",
             "- `compute_cycles` = the hardware `cycle_count` counter for one inference: all",
             "  layers run back to back on chip, weights already loaded.",
             "- `host_cycles_per_sample` = everything the host spends on one inference: write the",
             "  input features, start..done, handshake, read the final outputs (one value per cycle).",
             "- `layer_at_a_time` = the same inference done the way the previous engine had to:",
             "  only one layer's weights on chip, so the host reloads every layer and copies",
             "  activations between layers. Measured on the same RTL.", ""]

    if models:
        lines += ["## Models (NUM_MACS = 4)", "",
                  "| Model | Dataset | Architecture | Params | Samples | Float acc | INT8 acc (Python) "
                  "| RTL acc | Bit-exact | Compute cycles | Host cycles/inference | Layer-at-a-time | Speedup |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in models:
            lines.append(f"| {r['model']} | {r['dataset']} | {r['architecture']} | {r['parameters']} | "
                         f"{r['samples']} | {pct(r['float_accuracy'])} | {pct(r['quantized_accuracy'])} | "
                         f"{pct(r['rtl_accuracy'])} | {pct(r['bit_exact_rate'])} | {r['compute_cycles']} | "
                         f"{r['host_cycles_per_sample']} | {r['layer_at_a_time_host_cycles']} | "
                         f"{float(r['host_speedup']):.2f}x |")
        lines.append("")

    if scaling:
        lines += ["## MAC scaling (compute cycles per inference)", "",
                  "| Model | MACs | Compute cycles | Speedup vs 1 MAC | MAC utilization | MACs/cycle | RTL acc |",
                  "|---|---|---|---|---|---|---|"]
        for r in scaling:
            lines.append(f"| {r['model']} | {r['num_macs']} | {r['compute_cycles']} | "
                         f"{float(r['speedup_vs_1mac']):.2f}x | {pct(r['mac_utilization'])} | "
                         f"{float(r['macs_per_cycle']):.2f} | {pct(r['rtl_accuracy'])} |")
        lines += ["", "### Aggregate (one inference of every model)", "",
                  "| MACs | Total compute cycles | Speedup vs 1 MAC | Mean MAC utilization |",
                  "|---|---|---|---|"]
        totals = {n: sum(int(r["compute_cycles"]) for r in scaling if int(r["num_macs"]) == n)
                  for n in MAC_COUNTS}
        for n in MAC_COUNTS:
            utils = [float(r["mac_utilization"]) for r in scaling if int(r["num_macs"]) == n]
            lines.append(f"| {n} | {totals[n]} | {totals[1] / totals[n]:.2f}x | "
                         f"{sum(utils) / len(utils) * 100:.1f}% |")
        lines.append("")

    if len(baseline) == 2:
        old, new = baseline
        old_c, new_c = int(old["start_to_done"]), int(new["start_to_done"])
        lines += ["## Original design vs new design", "",
                  "Same workload (tiny model layer 0, 8 -> 16, run as a 1-layer network) on 4 MACs.",
                  "Latency is measured the same way",
                  "for both: clock edges from asserting `start` until `done` is seen.", "",
                  "| Design | Samples (bit-exact) | start->done cycles | MAC utilization |", "|---|---|---|---|",
                  f"| {old['design']} | {old['samples']} | {old_c} | {pct(old['mac_utilization'])} |",
                  f"| {new['design']} | {new['samples']} | {new_c} | {pct(new['mac_utilization'])} |", "",
                  f"Latency reduction: {(old_c - new_c) / old_c * 100:.1f}% ({old_c / new_c:.2f}x fewer cycles).", ""]

    if synth:
        lines += ["## Synthesis (Lattice ECP5 LFE5U-85F, Yosys + nextpnr, out-of-context)", "",
                  "| MACs | LUT4 | Flip-flops | DSP (MULT18X18D) | Block RAM (DP16KD) | LUT-RAM (DPR16X4) "
                  "| Logic cells after P&R | Fmax (MHz) | Critical path |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for n, r in synth.items():
            lines.append(f"| {n} | {r['lut4']} | {r['flip_flops']} | {r['dsp_mult18x18']} | "
                         f"{r['block_ram_dp16kd']} | {r['lutram_dpr16x4']} | {r['pnr_logic_cells_used']} | "
                         f"{r['fmax_mhz']} | {r['critical_path_ns']} ns: {r['critical_path_from']} -> "
                         f"{r['critical_path_to']} |")
        lines.append("")

    if synth and scaling:
        # DERIVED: measured cycles x measured post-route Fmax
        throughput = []
        for r in scaling:
            fmax = float(synth[r["num_macs"]]["fmax_mhz"]) * 1e6
            per_second = fmax / int(r["compute_cycles"])
            throughput.append({"model": r["model"], "num_macs": r["num_macs"],
                               "fmax_mhz": synth[r["num_macs"]]["fmax_mhz"],
                               "compute_cycles": r["compute_cycles"],
                               "inferences_per_second": round(per_second),
                               "latency_us": round(1e6 / per_second, 3),
                               "gmac_per_second": round(per_second * int(r["mac_operations"]) / 1e9, 4)})
        write_csv(RESULTS_DIR / "throughput.csv", throughput, list(throughput[0]))
        lines += ["## Throughput (DERIVED = post-route Fmax / measured compute cycles)", "",
                  "Assumes weights are already loaded and excludes host I/O.", "",
                  "| Model | MACs | Fmax (MHz) | Inferences/s | Latency (us) | GMAC/s |", "|---|---|---|---|---|---|"]
        for t in throughput:
            lines.append(f"| {t['model']} | {t['num_macs']} | {t['fmax_mhz']} | {t['inferences_per_second']:,} | "
                         f"{t['latency_us']} | {t['gmac_per_second']} |")
        lines.append("")

    (RESULTS_DIR / "summary.md").write_text("\n".join(lines))
    print("\n".join(lines))


def cmd_synth(args):
    run([sys.executable, PROJECT_ROOT / "synth" / "run_synth.py"])
    cmd_report(args)


def cmd_runtime(args):
    n = args.num_macs
    # GNU Make refuses to build in a directory whose path contains spaces, so
    # if the project lives in one, build in the system temp folder instead.
    obj_dir = BUILD_DIR / f"runtime_macs{n}"
    if " " in str(obj_dir):
        import tempfile
        obj_dir = Path(tempfile.gettempdir()) / "nn_accelerator_runtime" / f"macs{n}"
    shutil.rmtree(obj_dir, ignore_errors=True)
    obj_dir.mkdir(parents=True)
    shutil.copy(PROJECT_ROOT / "runtime" / "main.cpp", obj_dir / "main.cpp")
    run([verilator(), "--cc", "--exe", "-O2", f"-GNUM_MACS={n}", "-CFLAGS", f"-DNUM_MACS={n}",
         "--Mdir", obj_dir, *rtl_sources(), obj_dir / "main.cpp",
         "--top-module", "nn_accelerator_top"])
    # Verilator's makefile runs a small Python helper; point it at this Python.
    # -O2 instead of Verilator's default -Os: MSYS2's GCC 16.2 fails to link
    # std::string at -Os (undefined basic_string move constructor).
    run([make(), "-C", obj_dir, "-f", "Vnn_accelerator_top.mk", "-j4",
         f"PYTHON3={Path(sys.executable).as_posix()}",
         "OPT_FAST=-O2", "OPT_SLOW=-O2", "OPT_GLOBAL=-O2"])
    run([obj_dir / "Vnn_accelerator_top", PROJECT_ROOT / "compiled" / args.model])


def cmd_demo(args):
    model, n = args.model, args.num_macs
    if not (PROJECT_ROOT / "compiled" / model / "manifest.json").exists():
        cmd_train(args)
        cmd_compile(args)
    manifest = json.loads((PROJECT_ROOT / "compiled" / model / "manifest.json").read_text())
    info = json.loads((PROJECT_ROOT / "models" / model / "info.json").read_text())
    print(f"\nModel:   {model}  ({info['dataset']}, {info['architecture']}, "
          f"{info['num_parameters']} parameters)")
    for i, layer in enumerate(manifest["layers"]):
        print(f"  layer {i}: {layer['input_dim']:2d} -> {layer['output_dim']:2d}  "
              f"relu={layer['relu']}  shift={layer['output_shift']}")
    print(f"Hardware: NUM_MACS={n}\n")

    r = evaluate_model(model, simulate_model(model, n, args.max_samples))
    rtl = json.loads((BUILD_DIR / "results" / f"{model}_macs{n}.json").read_text())
    _, _, _, expected, _ = golden.load_compiled_model(PROJECT_ROOT / "compiled" / model)
    for i in range(min(10, r["samples"])):
        ok = rtl["rtl_outputs"][i] == expected[i].tolist()
        print(f"sample {i:3d}: label={r['labels'][i]}  float={r['float_predictions'][i]}  "
              f"golden INT8={r['quantized_predictions'][i]}  RTL={rtl['rtl_predictions'][i]}  "
              f"{'PASS' if ok else 'FAIL'}")
    print(f"... ({r['samples']} samples total)\n")
    print(f"Bit-exact samples:      {r['bit_exact_rate']*100:.1f}%  ({r['integer_mismatches']} integer mismatches)")
    print(f"Accuracy float/INT8/RTL: {r['float_accuracy']*100:.2f}% / "
          f"{r['quantized_accuracy']*100:.2f}% / {r['rtl_accuracy']*100:.2f}%")
    print(f"Compute cycles/inference: {r['compute_cycles']}  "
          f"(host-visible incl. I/O: {r['host_cycles_per_sample']})")
    print(f"MAC utilization:        {r['mac_utilization']*100:.1f}%   "
          f"MACs/cycle: {r['macs_per_cycle']:.2f}")
    print("RESULT:", "PASS" if r["bit_exact_rate"] == 1.0 else "FAIL")


def cmd_all(args):
    for command in [cmd_train, cmd_compile, cmd_lint, cmd_test, cmd_benchmark, cmd_synth, cmd_runtime]:
        command(args)


def main():
    commands = {"train": cmd_train, "compile": cmd_compile, "lint": cmd_lint, "test": cmd_test,
                "benchmark": cmd_benchmark, "synth": cmd_synth, "report": cmd_report, "runtime": cmd_runtime,
                "demo": cmd_demo, "all": cmd_all}
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=commands)
    parser.add_argument("--model", default="digits", choices=MODELS)
    parser.add_argument("--num-macs", type=int, default=4, choices=MAC_COUNTS)
    parser.add_argument("--max-samples", type=int, default=0, help="0 = every test sample")
    args = parser.parse_args()
    commands[args.command](args)


if __name__ == "__main__":
    main()
