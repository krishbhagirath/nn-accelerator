# INT8 Neural-Network Accelerator

A small, readable INT8 fully-connected neural-network accelerator in SystemVerilog,
with the full software stack around it:

```
PyTorch model (trained on a real dataset)
   ↓  torch.onnx.export
ONNX graph (Gemm / Relu only)
   ↓  python/compile_onnx.py
INT8 model: manifest.json + weight/bias text files
   ↓  sim/accelerator.py (cocotb)   or   runtime/main.cpp (Verilator C++)
SystemVerilog layer engine (rtl/), NUM_MACS = 1, 2, 4 or 8
   ↓
Prediction + hardware performance counters
```

The same RTL, compiler and pipeline run five different networks (4 datasets) and
are bit-exact against a Python integer golden model on every test sample.

## 1. What this project is

The accelerator is a **generic fully-connected layer engine**. Software runs a
network by invoking it once per layer, the way a driver launches a real
accelerator:

```
for each layer:
    load weights + biases + input activations
    set input_dim, output_dim, output_shift, relu
    pulse start, wait for done
    read the INT8 outputs  -> they become the next layer's inputs
prediction = argmax(final outputs)
```

`NUM_MACS` (how many neurons are computed in parallel) is a compile-time
parameter. Layer sizes, shift and ReLU are run-time settings, up to
`MAX_INPUTS = 64` and `MAX_OUTPUTS = 32`.

## 2. Architecture

```
               host writes                          host reads
                   │                                    ▲
   ┌───────────────┼────────────────────────────────────┼──────────────┐
   │  input memory (64 x INT8, shared)                  │              │
   │        │ input[i]                                  │              │
   │        ├──► lane 0: weight bank 0 ─► MAC ─► shift/clamp ─┐        │
   │        ├──► lane 1: weight bank 1 ─► MAC ─► shift/clamp ─┼► output│
   │        ├──► ...                                          │  memory│
   │        └──► lane N-1 ...                                 ┘ (32x8) │
   │  controller (IDLE / COMPUTE / DRAIN / DONE)   perf counters       │
   └───────────────────────────────────────────────────────────────────┘
```

| File | What it does |
|---|---|
| `rtl/nn_accelerator_top.sv` | Wires everything; `generate` loop creates `NUM_MACS` lanes |
| `rtl/controller.sv` | Walks (batch, input) pairs; 3-stage pipeline control |
| `rtl/mac_unit.sv` | `acc = (first ? bias : acc) + input * weight` |
| `rtl/post_process.sv` | `acc >>> shift`, then clamp to 0..127 (ReLU) or -128..127 |
| `rtl/sync_ram.sv` | Memory with 1-cycle read latency (maps to FPGA block RAM) |
| `rtl/output_memory.sv` | Stores a layer's outputs, `NUM_MACS` writes per batch |
| `rtl/perf_counters.sv` | `cycle_count`, `mac_active_cycles`, `mac_operations` |
| `baseline/rtl/` | The original fixed 8→16, 4-MAC design, kept for comparison |

## 3. Hardware datapath

**Batching across lanes.** Neuron `n` is computed by lane `n % NUM_MACS`. Batch
`b` computes neurons `b*NUM_MACS ... b*NUM_MACS + NUM_MACS-1`. A layer needs
`ceil(output_dim / NUM_MACS)` batches. In the last batch, unused lanes are
masked off: they are never written and never counted.

**Weight banks.** Each lane has its own weight memory holding only its neurons:
`bank address = (n / NUM_MACS) * MAX_INPUTS + input_index`. All lanes read the
same address each cycle, so one counter drives everything and every bank needs
only one read port.

**Accumulator starts at the bias.** On a neuron's first input,
`acc = bias + x*w`. That removes the separate "clear" and "add bias" steps.

**3-stage pipeline.** The memories take one cycle to return data:

| Stage | What happens |
|---|---|
| 1 READ | controller presents (batch, input_index) to the memories |
| 2 MAC | data arrives; every lane accumulates |
| 3 WRITE | after a neuron's last input, results go to output memory |

The control flags are the stage-1 conditions delayed by one or two cycles. Batch
`b+1` starts reading while batch `b` is being written, so there are no idle
cycles between batches:

```
cycle_count = ceil(output_dim / NUM_MACS) * input_dim + 2
```

The cocotb tests assert this formula exactly.

**Arithmetic.**
- INT8 × INT8 products and an INT32 accumulator.
- Worst case for a 64-input layer is about 2^20, far from overflow.
- `>>>` (round toward −∞), then clamp.

## 4. Software / model pipeline

| Step | File | Output |
|---|---|---|
| Train + export | `python/train_model.py` | `models/<name>/model.onnx`, float calibration/test data |
| Compile | `python/compile_onnx.py` | `compiled/<name>/manifest.json`, `layer_<i>_weights.txt`, `layer_<i>_biases.txt`, `test_inputs.txt`, `expected_outputs.txt` |
| Golden model | `python/golden.py` | the ONE integer implementation of the hardware arithmetic |
| Run on RTL | `sim/test_model.py` via `run_pipeline.py` | per-sample outputs + counters |

The compiler walks the ONNX graph and accepts only `Gemm` and `Relu`. It
rejects anything else with a clear error.

## 5. Quantization

All of it lives in `compile_onnx.py`, and is deliberately simple:

- **Weights:** symmetric per-tensor INT8, `w_q = round(w * 127 / max|w|)`.
- **Inputs:** `x_q = round(x * 127 / max|calibration x|)`, clipped to ±127.
- **Biases:** INT32 at the accumulator's scale, `input_scale * weight_scale`.
- **Requantization:** `out = clamp(acc >> shift)`.
  - The compiler picks the **smallest shift** for which the largest calibration
    accumulator still fits in INT8.
  - That keeps the most precision with no saturation on calibration data.
  - The next layer's input scale is `accumulator_scale / 2^shift`.
- **Hidden layers:** ReLU, clamp 0..127.
- **Final layer:** signed logits, clamp −128..127.

## 6. Verification

Every test asserts its result automatically. Nothing relies on reading simulator
output by eye.

- **`tests/test_python.py`** (pytest, 22 tests):
  - golden arithmetic (floor shift, clamps)
  - shift selection
  - the compiler accepting `Gemm`/`Relu` and rejecting unsupported ops
  - ONNX checker on every model
  - onnxruntime vs our reading of the graph
  - compiled files consistent with the golden model
- **`sim/test_accelerator.py`** (cocotb, 7 tests, run for NUM_MACS = 1, 2, 4, 8):

| Test | What it checks |
|---|---|
| `test_reset_state` | idle and zeroed counters after reset |
| `test_start_done_handshake` | busy/done behavior; config captured at start; rerun without reset |
| `test_reset_during_inference` | reset mid-layer returns to idle, and the next run is correct |
| `test_random_layers` | 40 random layers: sizes 1×1 to 64×32, non-multiples of NUM_MACS, full-range INT8, random shift/ReLU |
| `test_extreme_values` | ±128 extremes, zeros, saturation both ways, ±2³¹ biases, shift 31, exact clamp boundaries |
| `test_cycle_counts` | counters equal the formula above for 7 shapes |
| `test_compiled_models` | 20 samples of every compiled model, every layer bit-exact |

- **`sim/test_model.py`:** full-model run of every test sample (used by the
  benchmark and demo).
- **`sim/test_baseline.py`:** measures the original design on the same workload.
- **`runtime/main.cpp`:** the C++ Verilator runtime independently re-checks
  bit-exactness.

## 7. Running the project

```
pip install -r requirements.txt

python run_pipeline.py train       # train 5 models, export ONNX
python run_pipeline.py compile     # ONNX -> INT8 files
python run_pipeline.py lint        # Verilator -Wall, NUM_MACS = 1/2/4/8
python run_pipeline.py test        # pytest + cocotb regression for 1/2/4/8 MACs
python run_pipeline.py benchmark   # all models x all MAC counts + original design -> results/
python run_pipeline.py synth       # Yosys + nextpnr for 1/2/4/8 MACs -> results/
python run_pipeline.py runtime --model digits --num-macs 4   # C++ Verilator runtime
python run_pipeline.py demo    --model digits --num-macs 4   # one model, readable output
python run_pipeline.py all
```

**Tools.** On Windows, install the HDL tools with MSYS2 (UCRT64):

```
pacman -S mingw-w64-ucrt-x86_64-iverilog mingw-w64-ucrt-x86_64-verilator mingw-w64-ucrt-x86_64-gcc mingw-w64-ucrt-x86_64-make mingw-w64-ucrt-x86_64-yosys mingw-w64-ucrt-x86_64-nextpnr mingw-w64-ucrt-x86_64-prjtrellis
```

`run_pipeline.py` puts `C:\msys64\ucrt64\bin` first on PATH automatically. It
also works around three Windows issues:
- `verilator` is a Perl script, so it calls `verilator_bin` directly.
- Make can't build in paths with spaces, so the runtime builds in the temp folder.
- MSYS2 GCC 16.2 fails to link `std::string` at `-Os`, so the runtime builds with `-O2`.

On Linux: `apt install iverilog verilator yosys nextpnr-ecp5`.

## 8. Supported models

| Model | Dataset | Layers | Parameters |
|---|---|---|---|
| tiny | synthetic 8-feature, 2 classes (the original project model) | 8 → 16 → 2 | 178 |
| iris | sklearn Iris | 4 → 8 → 3 | 67 |
| cancer | sklearn Breast Cancer Wisconsin | 30 → 16 → 2 | 530 |
| digits | sklearn Digits (8×8 handwritten digits) | 64 → 16 → 10 | 1210 |
| digits_deep | sklearn Digits | 64 → 32 → 16 → 10 | 2778 |

- **Data splits:** sklearn datasets use a stratified 80/20 split (`random_state=0`).
- **Input scaling:** Iris and Cancer features are standardized with training
  statistics; Digits pixels are divided by 16. This happens in software before
  the accelerator, like image normalization.
- **Pretrained models:** none from a public model hub are used. Every model is
  trained locally.

## 9. Results

All numbers below are copied from `results/summary.md`, which is generated by
`run_pipeline.py`. They are measured in RTL simulation unless marked otherwise.

**Per model, at NUM_MACS = 4:**

| Model | Samples | Float acc | INT8 acc (Python) | RTL acc | Bit-exact | Compute cycles |
|---|---|---|---|---|---|---|
| tiny | 400 | 99.75% | 99.75% | 99.75% | 100% | 52 |
| iris | 30 | 100.00% | 100.00% | 100.00% | 100% | 20 |
| cancer | 114 | 96.49% | 96.49% | 96.49% | 100% | 140 |
| digits | 360 | 96.67% | 96.67% | 96.67% | 100% | 308 |
| digits_deep | 360 | 96.11% | 96.11% | 96.11% | 100% | 694 |

- **Bit-exactness:** 1,264 test samples per MAC configuration, 0 integer
  mismatches, for all 4 configurations.
- **Quantization effect:** INT8 accuracy equals float accuracy on these test
  sets, but quantization does change 3 individual predictions (2 of 114 Cancer
  samples, 1 of 360 Digits samples). The changes cancel out in the totals. With
  more data or harder models, expect a real accuracy drop.

**Original vs new design** (same workload: tiny layer 0, 8→16, 4 MACs, measured
identically from start to done):

| Design | start→done cycles | MAC utilization | Layers it can run |
|---|---|---|---|
| original (`baseline/rtl`) | 49 | 65.3% | only 8→16 + ReLU |
| new engine | 35 | 91.4% | any layer up to 64→32, ReLU or signed output |

That's a 28.6% latency reduction. It comes from:
- loading the accumulator with the bias (removes CLEAR);
- removing the idle POST_PROCESS and NEXT_BATCH states;
- overlapping each batch's write with the next batch's reads.

## 10. MAC scaling

Compute cycles for one inference (sum of the hardware `cycle_count` over all layers):

| Model | 1 MAC | 2 MACs | 4 MACs | 8 MACs | Speedup 1→8 | Utilization at 8 |
|---|---|---|---|---|---|---|
| tiny | 164 | 84 | 52 | 36 | 4.56× | 55.6% |
| iris | 60 | 36 | 20 | 16 | 3.75× | 43.8% |
| cancer | 516 | 260 | 140 | 80 | 6.45× | 80.0% |
| digits | 1188 | 596 | 308 | 164 | 7.24× | 90.2% |
| digits_deep | 2726 | 1366 | 694 | 358 | 7.61× | 95.0% |
| **all models** | 4654 | 2342 | 1214 | 654 | **7.12×** | 72.9% (mean) |

Speedup is close to linear for large layers. Small layers lose efficiency for
two reasons:
- the fixed 2-cycle pipeline drain per layer;
- partial batches. For example, a 2-output layer uses 2 of 8 lanes.

## 11. Synthesis results

**Flow:**
- Lattice ECP5 LFE5U-85F, speed grade 6, using Yosys `synth_ecp5` then
  `nextpnr-ecp5`.
- Place-and-route runs out of context (no package pins) with fixed seed 1.
- Fmax is nextpnr's post-route timing estimate.

**How the memories mapped:**
- Weight banks → DP16KD block RAM.
- Bias banks and the input memory → LUT-RAM.
- Output memory (asynchronous read) → flip-flops.

| MACs | LUT4 | FF | DSP | BRAM | LUT-RAM | Logic cells (P&R) | Fmax |
|---|---|---|---|---|---|---|---|
| 1 | 512 | 249 | 1 | 1 | 28 | 952 | 61.05 MHz |
| 2 | 1098 | 566 | 2 | 2 | 24 | 1622 | 60.03 MHz |
| 4 | 1859 | 693 | 4 | 4 | 40 | 2703 | 60.79 MHz |
| 8 | 3445 | 952 | 8 | 8 | 72 | 4945 | 57.11 MHz |

- **Critical path (all variants):** block-RAM read → multiply → 32-bit
  accumulate, about 16.4–17.5 ns, in a single cycle.
- **Throughput** (DERIVED = Fmax ÷ measured cycles; weights resident, no host
  I/O), for digits:
  - 51k inferences/s at 1 MAC (0.061 GMAC/s);
  - 348k inferences/s at 8 MACs (0.41 GMAC/s).
- **Not available:** ASIC area and power. No standard-cell library was used.

## 12. Limitations

- **Fully-connected layers only** (Gemm + ReLU). No convolution, pooling or
  other activations.
- **Size limits:** at most 64 inputs and 32 outputs per layer; NUM_MACS must be
  a power of two.
- **Simple quantization:**
  - per-tensor scales only;
  - power-of-two rescaling only;
  - the shift rounds toward −∞ instead of to nearest;
  - the input scale is set by the single largest calibration value.
- **Host I/O dominates small models.** The host writes one value per cycle, and
  weights must be reloaded for every layer because only one layer fits in the
  weight memory.
- **Fmax is limited** by the single-cycle read → multiply → accumulate path.
- **Interface:** the host interface is direct ports, not a standard bus
  (AXI/Wishbone).
- **CI** (`.github/workflows/ci.yml`) is written but has not been run on GitHub yet.

## 13. Future work

- Register the multiplier output (a 4-stage pipeline) to raise Fmax.
- Round-to-nearest requantization and per-channel weight scales.
- A wider host write port, or double-buffered weight memory, to hide load time.
- A 4×4 systolic array to compare against the current lane-parallel design.
- Convolution expressed as matrix multiply (im2col) through the same engine.
