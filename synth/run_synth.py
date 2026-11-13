# Synthesizes the accelerator for NUM_MACS = 1, 2, 4, 8 and collects metrics.
#
#   1. Yosys  (synth.ys): RTL -> ECP5 cells. Gives LUT / FF / DSP / RAM counts.
#   2. nextpnr-ecp5:      place & route on an LFE5U-85F (speed grade 6).
#                         Gives post-route utilization and maximum frequency.
#
# nextpnr is run "out of context": no I/O pins are assigned, because the
# accelerator's host interface (~200 port bits) is meant to connect to on-chip
# logic, not to package pins. Fmax is therefore the internal register-to-
# register speed of the accelerator.
#
# Output: results/synthesis_results.csv

import csv
import json
import re
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BUILD = PROJECT_ROOT / "build" / "synth"
RESULTS = PROJECT_ROOT / "results"
MAC_COUNTS = [1, 2, 4, 8]
SEED = 1                 # fixed so results are reproducible
TARGET_MHZ = 100         # nextpnr optimizes toward this; achieved Fmax is reported


def yosys_cell_counts(stat_file):
    """Parse 'stat' output lines like '     1859   LUT4'."""
    counts = {}
    for line in stat_file.read_text().splitlines():
        match = re.match(r"\s+(\d+)\s+(\$?[A-Za-z_][\w$]*)\s*$", line)
        if match:
            counts[match.group(2)] = int(match.group(1))
    return counts


def synthesize(num_macs):
    # Relative to the project root: Yosys scripts can't handle spaces in paths.
    out = Path("build") / "synth" / f"macs{num_macs}"
    script = (PROJECT_ROOT / "synth" / "synth.ys").read_text()
    script = script.replace("{NUM_MACS}", str(num_macs)).replace("{OUT}", out.as_posix())
    script_file = BUILD / f"synth_macs{num_macs}.ys"
    script_file.write_text(script)

    print(f"[synth] yosys   NUM_MACS={num_macs}")
    subprocess.run(["yosys", "-q", "-s", script_file], cwd=PROJECT_ROOT, check=True)
    cells = yosys_cell_counts(PROJECT_ROOT / f"{out}_stat.txt")

    print(f"[synth] nextpnr NUM_MACS={num_macs}")
    report_file = PROJECT_ROOT / f"{out}_pnr.json"
    subprocess.run(["nextpnr-ecp5", "--85k", "--package", "CABGA381", "--speed", "6",
                    "--json", PROJECT_ROOT / f"{out}.json", "--out-of-context", "--freq", str(TARGET_MHZ),
                    "--seed", str(SEED), "--timing-allow-fail", "--report", report_file],
                   check=True, capture_output=True)
    report = json.loads(report_file.read_text())
    used = {name: v["used"] for name, v in report["utilization"].items()}

    # The slowest register-to-register path: which cell it starts and ends at.
    path = report["critical_paths"][0]["path"]
    critical_start = path[0]["from"]["cell"]
    critical_end = path[-1]["to"]["cell"]
    critical_delay = sum(step["delay"] for step in path)
    return {
        "num_macs": num_macs,
        "yosys_total_cells": cells.get("cells", sum(v for k, v in cells.items() if not k.startswith("$"))),
        "lut4": cells.get("LUT4", 0),
        "carry_cells_ccu2c": cells.get("CCU2C", 0),
        "flip_flops": cells.get("TRELLIS_FF", 0),
        "dsp_mult18x18": cells.get("MULT18X18D", 0),
        "block_ram_dp16kd": cells.get("DP16KD", 0),
        "lutram_dpr16x4": cells.get("TRELLIS_DPR16X4", 0),
        "pnr_logic_cells_used": used.get("TRELLIS_COMB", 0),
        "pnr_logic_cells_available": report["utilization"]["TRELLIS_COMB"]["available"],
        "fmax_mhz": round(report["fmax"]["clk"]["achieved"], 2),
        "critical_path_ns": round(critical_delay, 2),
        "critical_path_from": critical_start,
        "critical_path_to": critical_end,
    }


def main():
    BUILD.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(exist_ok=True)
    rows = [synthesize(n) for n in MAC_COUNTS]
    with open(RESULTS / "synthesis_results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
