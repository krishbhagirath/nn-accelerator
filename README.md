# INT8 Neural-Network Accelerator

A small, readable INT8 neural-network accelerator in SystemVerilog that runs a
**whole fully-connected network on chip from one start pulse**, with the full
software stack around it:

```
PyTorch model (trained on a real dataset)
   ↓  torch.onnx.export
ONNX graph (Gemm / Relu only)
   ↓  python/compile_onnx.py
INT8 model: manifest.json + weight/bias text files
   ↓  sim/accelerator.py (cocotb)   or   runtime/main.cpp (Verilator C++)
SystemVerilog accelerator (rtl/), NUM_MACS = 1, 2, 4 or 8
   ↓
Prediction + hardware performance counters
```

The same RTL, compiler and pipeline run five different networks (4 datasets) and
are bit-exact against a Python integer golden model on every test sample.

## 1. What this project is

The host loads a model **once**. After that, every inference is just
"inputs in, outputs out":

```
once per model:
    write the layer table (dims, shift, relu for each layer)
    write every layer's weights and biases          -> they stay on chip
once per inference:
    write the input features
    set num_layers, pulse start, wait for done      -> all layers run on chip
    read the final outputs
prediction = argmax(outputs)
```

The host never sees the hidden-layer activations: each layer's outputs land
directly where the next layer reads its inputs.

- `NUM_MACS` (how many neurons are computed in parallel) is a compile-time parameter.
- Up to `MAX_LAYERS = 4` layers, each up to `MAX_INPUTS = 64` inputs and
  `MAX_OUTPUTS = 32` outputs. Layer sizes, shifts and ReLU are run-time settings.

## 2. One inference, end to end

This is everything that happens for one digits sample (64 → 16 → 10) on 4 MACs.
The cycle numbers are measured (see `results/`).

| Step | Who | What happens | Cycles |
|---|---|---|---|
| 0 | host, once | write 2 layer-table entries + 1,210 weights/biases | 1,212 (one time) |
| 1 | host | write 64 input pixels into the current activation buffer | 64 |
| 2 | host | set `config_num_layers = 2`, raise `start` | |
| 3 | controller | layer 0: 4 batches × 64 inputs, outputs → other buffer | 258 |
| 4 | controller | flip buffers, load table entry 1 — same cycle, no gap | 0 |
| 5 | controller | layer 1: 3 batches × 16 inputs, outputs → first buffer | 50 |
| 6 | controller | flip buffers, go to DONE | |
| 7 | host | see `done`, drop `start`, read the 10 outputs, argmax | ~13 |
| | | **total per inference** | **385** |

Doing the same inference layer-at-a-time (how the previous version of this
engine had to work, with only one layer's weights on chip) costs **1,632**
cycles, because the host reloads every layer's weights and copies activations
in and out between layers. Running the network on chip is **4.24× less host
time** for the same answer.

## 3. Architecture

```
                 host writes                                         host reads
                     │                                                   ▲
 ┌───────────────────┼───────────────────────────────────────────────────┼─────┐
 │ layer table (dims, shift, relu per layer) ──► controller ──► perf counters │
 │                                                  │                         │
 │  activation buffer A ◄─┐ (ping-pong: swap        │ (layer, batch, input)   │
 │  activation buffer B ◄─┤  after every layer)     ▼                         │
 │        │ input[i]      │                                                   │
 │        ├──► lane 0: weight bank 0 ─► MAC ─► shift/clamp ─┐                 │
 │        ├──► lane 1: weight bank 1 ─► MAC ─► shift/clamp ─┼─► other buffer │
 │        └──► lane N-1 ...                                 ┘                 │
 └─────────────────────────────────────────────────────────────────────────────┘
```

| File | What it does |
|---|---|
| `rtl/nn_accelerator_top.sv` | Wires everything; `generate` loop creates `NUM_MACS` lanes and the two buffers |
| `rtl/controller.sv` | Walks (layer, batch, input); 3-stage pipeline control; moves to the next layer |
| `rtl/layer_table.sv` | One entry per layer: input_dim, output_dim, shift, relu |
| `rtl/activation_buffer.sv` | One activation vector, stored as rows of `NUM_MACS` values |
| `rtl/mac_unit.sv` | `acc = (first ? bias : acc) + input * weight` |
| `rtl/post_process.sv` | `acc >>> shift`, then clamp to 0..127 (ReLU) or -128..127 |
| `rtl/sync_ram.sv` | Memory with 1-cycle read latency (maps to FPGA block RAM) |
| `rtl/perf_counters.sv` | `cycle_count`, `mac_active_cycles`, `mac_operations` for the whole run |
| `baseline/rtl/` | The original fixed 8→16, 4-MAC design, kept for comparison |

## 4. Hardware datapath

**Batching across lanes.** Neuron `n` is computed by lane `n % NUM_MACS`. Batch
`b` computes neurons `b*NUM_MACS ... b*NUM_MACS + NUM_MACS-1`. A layer needs
`ceil(output_dim / NUM_MACS)` batches. In the last batch, unused lanes are
masked off: they are never written and never counted.

**Weight banks with a slot per layer.** Each lane has its own weight memory
holding only its neurons, for every layer:

```
row          = layer * MAX_BATCHES + (n / NUM_MACS)
bank address = row * MAX_INPUTS + input_index
```

All lanes read the same address each cycle, so one counter drives everything
and every bank needs only one read port. Each layer has a fixed slot, which
wastes some space for small layers but keeps the addressing to one line.

**Ping-pong activation buffers.** There are two activation buffers. A layer
reads its inputs from one and writes its outputs into the other; then they swap
(`buffer_select` flips). So a layer's outputs are already in place as the next
layer's inputs, and nothing is copied. The host always talks to the *current*
buffer: before `start` it holds the inputs, after `done` it holds the outputs.

Each buffer is stored as rows of `NUM_MACS` values, because that is exactly how
results arrive: batch `b` writes row `b`, lane `k` writes column `k`. That keeps
the write logic small: an earlier version where any lane could write any address
needed 2.3× the LUTs at 4 MACs (4,474 vs 1,905).

**Accumulator starts at the bias.** On a neuron's first input,
`acc = bias + x*w`. That removes the separate "clear" and "add bias" steps.

**3-stage pipeline.** The memories take one cycle to return data:

| Stage | What happens |
|---|---|
| 1 READ | controller presents (layer, batch, input_index) to the memories |
| 2 MAC | data arrives; every lane accumulates |
| 3 WRITE | after a neuron's last input, results go to the other buffer |

The control flags are the stage-1 conditions delayed by one or two cycles. Batch
`b+1` starts reading while batch `b` is being written, so there are no idle
cycles between batches. Between layers the controller waits 2 cycles for the
last batch to be written, and on that same cycle flips the buffers and loads the
next layer, so:

```
cycle_count = sum over layers of ( ceil(output_dim / NUM_MACS) * input_dim + 2 )
```

That is exactly the same compute time as running each layer separately: moving
to the next layer costs nothing extra. The cocotb tests assert this formula.

**Arithmetic.**
- INT8 × INT8 products and an INT32 accumulator.
- Worst case for a 64-input layer is about 2^20, far from overflow.
- `>>>` (round toward −∞), then clamp.

## 5. Software / model pipeline

| Step | File | Output |
|---|---|---|
| Train + export | `python/train_model.py` | `models/<name>/model.onnx`, float calibration/test data |
| Compile | `python/compile_onnx.py` | `compiled/<name>/manifest.json`, `layer_<i>_weights.txt`, `layer_<i>_biases.txt`, `test_inputs.txt`, `expected_outputs.txt` |
| Golden model | `python/golden.py` | the ONE integer implementation of the hardware arithmetic |
| Run on RTL | `sim/test_model.py` via `run_pipeline.py` | per-sample outputs + counters |

The compiler walks the ONNX graph and accepts only `Gemm` and `Relu`. It
rejects anything else, layers over 64→32, and models over 4 layers, with a
clear error.

## 6. Quantization

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

## 7. Verification

Every test asserts its result automatically. Nothing relies on reading simulator
output by eye.

- **`tests/test_python.py`** (pytest, 22 tests):
  - golden arithmetic (floor shift, clamps)
  - shift selection
  - the compiler accepting `Gemm`/`Relu` and rejecting unsupported ops
  - ONNX checker on every model
  - onnxruntime vs our reading of the graph
  - compiled files consistent with the golden model
- **`sim/test_accelerator.py`** (cocotb, 9 tests, run for NUM_MACS = 1, 2, 4, 8):

| Test | What it checks |
|---|---|
| `test_reset_state` | idle and zeroed counters after reset |
| `test_start_done_handshake` | busy/done behavior; config captured at start; rerun without reset |
| `test_reset_during_inference` | reset mid-layer returns to idle, and the next run is correct |
| `test_random_layers` | 40 random single layers: sizes 1×1 to 64×32, non-multiples of NUM_MACS, full-range INT8, random shift/ReLU |
| `test_extreme_values` | ±128 extremes, zeros, saturation both ways, ±2³¹ biases, shift 31, exact clamp boundaries |
| `test_cycle_counts` | counters equal the formula above for 7 shapes |
| `test_random_networks` | 24 random 1–4 layer networks, 3 back-to-back inferences each (checks the buffer swap for odd and even layer counts) |
| `test_max_size_network` | 4 layers, 64→32→32→32→32 |
| `test_compiled_models` | 20 samples of every compiled model, whole network on chip, bit-exact |

- **`sim/test_model.py`:** every test sample of a model (used by the benchmark
  and demo), plus the layer-at-a-time comparison.
- **`sim/test_baseline.py`:** measures the original design on the same workload.
- **`runtime/main.cpp`:** the C++ Verilator runtime independently re-checks
  bit-exactness.

To check that the tests can actually catch bugs, a deliberately broken
controller (buffers never swap) was run against them: 7 of the 9 tests failed.

## 8. Running the project

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

## 9. Supported models

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

## 10. Results

All numbers below are copied from `results/summary.md`, which is generated by
`run_pipeline.py`. They are measured in RTL simulation unless marked otherwise.

**Per model, at NUM_MACS = 4:**

| Model | Samples | Float acc | INT8 acc (Python) | RTL acc | Bit-exact | Compute cycles | Host cycles / inference | Layer-at-a-time | Less host time |
|---|---|---|---|---|---|---|---|---|---|
| tiny | 400 | 99.75% | 99.75% | 99.75% | 100% | 52 | 65 | 280 | 4.31× |
| iris | 30 | 100.00% | 100.00% | 100.00% | 100% | 20 | 30 | 118 | 3.93× |
| cancer | 114 | 96.49% | 96.49% | 96.49% | 100% | 140 | 175 | 742 | 4.24× |
| digits | 360 | 96.67% | 96.67% | 96.67% | 100% | 308 | 385 | 1632 | 4.24× |
| digits_deep | 360 | 96.11% | 96.11% | 96.11% | 100% | 694 | 771 | 3654 | 4.74× |

- **Host cycles / inference:** write inputs + start..done + read final outputs,
  with all weights already on chip.
- **Layer-at-a-time:** the same inference on the same RTL, but done the way the
  previous engine had to (reload each layer's weights, copy activations through
  the host). The gap grows with more MACs, because compute shrinks while the
  weight reloads don't: at 8 MACs digits_deep is **7.63×** less host time.
- **Bit-exactness:** 1,264 test samples per MAC configuration, 0 integer
  mismatches, for all 4 configurations.
- **Quantization effect:** INT8 accuracy equals float accuracy on these test
  sets, but quantization does change 3 individual predictions (2 of 114 Cancer
  samples, 1 of 360 Digits samples). The changes cancel out in the totals. With
  more data or harder models, expect a real accuracy drop.

**Original vs new design** (same workload: tiny layer 0, 8→16, 4 MACs, measured
identically from start to done):

| Design | start→done cycles | MAC utilization | What it can run |
|---|---|---|---|
| original (`baseline/rtl`) | 49 | 65.3% | only 8→16 + ReLU |
| new engine | 35 | 91.4% | any network up to 4 layers of 64→32, ReLU or signed output |

That's a 28.6% latency reduction. It comes from:
- loading the accumulator with the bias (removes CLEAR);
- removing the idle POST_PROCESS and NEXT_BATCH states;
- overlapping each batch's write with the next batch's reads.

## 11. MAC scaling

Compute cycles for one inference (the hardware `cycle_count`, all layers):

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

## 12. Synthesis results

**Flow:**
- Lattice ECP5 LFE5U-85F, speed grade 6, using Yosys `synth_ecp5` then
  `nextpnr-ecp5`.
- Place-and-route runs out of context (no package pins) with fixed seed 1.
- Fmax is nextpnr's post-route timing estimate.

**How the memories mapped:**
- Weight banks (4 layer slots each) → DP16KD block RAM.
- Bias banks → LUT-RAM.
- Activation buffers → LUT-RAM and/or flip-flops; Yosys decides per build. At
  8 MACs (8 writes per cycle) LUT-RAM use drops and the flip-flop count jumps,
  which suggests the buffers moved to flip-flops there (not confirmed per module).

| MACs | LUT4 | FF | DSP | BRAM | LUT-RAM | Logic cells (P&R) | Fmax |
|---|---|---|---|---|---|---|---|
| 1 | 619 | 309 | 1 | 5 | 38 | 1115 | 60.24 MHz |
| 2 | 1086 | 344 | 2 | 6 | 134 | 2270 | 59.00 MHz |
| 4 | 1905 | 460 | 4 | 4 | 198 | 3693 | 58.60 MHz |
| 8 | 4768 | 1742 | 8 | 8 | 70 | 6244 | 58.58 MHz |

- **Critical path (all variants):** block-RAM read → multiply → 32-bit
  accumulate, about 16.6–17.1 ns, in a single cycle.
- **Cost of whole-network execution:** compared with the single-layer engine
  (2,703 logic cells, 60.79 MHz at 4 MACs), holding 4 layers of weights and two
  activation buffers costs about +37% logic cells and −3.6% Fmax at 4 MACs.
  That is still 4.4% of this FPGA.
- **Throughput** (DERIVED = Fmax ÷ measured compute cycles; weights resident,
  no host I/O), for digits:
  - 50.7k inferences/s at 1 MAC (0.060 GMAC/s);
  - 357k inferences/s at 8 MACs (0.42 GMAC/s).
  - Every model is in `results/summary.md`.
- **Not available:** ASIC area and power. No standard-cell library was used.

## 13. Limitations

- **Fully-connected layers only** (Gemm + ReLU). No convolution, pooling or
  other activations.
- **Size limits:** at most 4 layers, 64 inputs and 32 outputs per layer;
  NUM_MACS must be a power of two. Each layer gets a fixed weight slot, so
  small layers leave memory unused.
- **Simple quantization:**
  - per-tensor scales only;
  - power-of-two rescaling only;
  - the shift rounds toward −∞ instead of to nearest;
  - the input scale is set by the single largest calibration value.
- **Host writes one value per cycle.** For small models, writing the inputs is
  still a noticeable part of each inference (64 of 385 cycles for digits).
- **Fmax is limited** by the single-cycle read → multiply → accumulate path.
- **Interface:** the host interface is direct ports, not a standard bus
  (Avalon/AXI/Wishbone).
- **Not run on a physical FPGA yet.** All numbers are RTL simulation or
  post-route estimates.

## 14. Future work

- Run on a DE10 board (Intel FPGA): a thin board wrapper with a bus interface
  (Avalon-MM) and measured on-board latency.
- Register the multiplier output (a 4-stage pipeline) to raise Fmax.
- Round-to-nearest requantization and per-channel weight scales.
- A wider host write port to cut input-loading time.
- A 4×4 systolic array to compare against the current lane-parallel design.
- Convolution expressed as matrix multiply (im2col) through the same engine.
