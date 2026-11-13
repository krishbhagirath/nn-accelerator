// Generic INT8 fully-connected layer engine.
//
// The host (cocotb testbench or C++ runtime) runs a whole network by calling
// this engine once per layer:
//     load weights + biases + inputs -> set config -> start -> wait done -> read outputs
// and then feeding those outputs in as the next layer's inputs.
//
// NUM_MACS (compile time) = how many neurons are computed in parallel.
// input_dim / output_dim / shift / ReLU (run time) = which layer is computed.
//
//   input memory --input[i]--+--> lane 0: weight bank 0, bias bank 0, MAC, post-process --+
//   (shared by all lanes)    +--> lane 1: weight bank 1, bias bank 1, MAC, post-process --+--> output memory
//                            +--> ...                                                     |
//                            +--> lane NUM_MACS-1 ----------------------------------------+
//
// Weight storage: neuron n lives in lane (n % NUM_MACS), row (n / NUM_MACS).
// Each lane has its own weight bank holding its neurons' weights, one row of
// MAX_INPUTS entries per neuron:
//     bank address = row * MAX_INPUTS + input_index
// During compute every lane reads the SAME address (row = current batch), so a
// single address counter feeds all lanes, and each bank needs only one read port.
// NUM_MACS must be a power of two (1, 2, 4, 8, ...) so % and / are just bit slices.

`timescale 1ns/1ps

module nn_accelerator_top #(
    parameter NUM_MACS    = 4,
    parameter MAX_INPUTS  = 64,
    parameter MAX_OUTPUTS = 32,

    localparam IN_DIM_W    = $clog2(MAX_INPUTS + 1),   // holds 0..MAX_INPUTS
    localparam IN_INDEX_W  = $clog2(MAX_INPUTS),       // holds 0..MAX_INPUTS-1
    localparam OUT_DIM_W   = $clog2(MAX_OUTPUTS + 1),
    localparam OUT_INDEX_W = $clog2(MAX_OUTPUTS),
    localparam MAX_BATCHES = (MAX_OUTPUTS + NUM_MACS - 1) / NUM_MACS,
    localparam BATCH_W     = (MAX_BATCHES > 1) ? $clog2(MAX_BATCHES) : 1,
    localparam BANK_DEPTH  = MAX_BATCHES * MAX_INPUTS,
    localparam BANK_ADDR_W = $clog2(BANK_DEPTH)
)(
    input  logic                    clk,
    input  logic                    reset,
    input  logic                    start,

    // ---------------- layer configuration ----------------
    input  logic [IN_DIM_W-1:0]     config_input_dim,
    input  logic [OUT_DIM_W-1:0]    config_output_dim,
    input  logic [4:0]              config_output_shift,
    input  logic                    config_relu,

    // ---------------- host: load input activations ----------------
    input  logic                    input_write_enable,
    input  logic [IN_INDEX_W-1:0]   input_write_address,
    input  logic signed [7:0]       input_write_data,

    // ---------------- host: load weights ----------------
    // weight[neuron][input] = data
    input  logic                    weight_write_enable,
    input  logic [OUT_INDEX_W-1:0]  weight_write_neuron,
    input  logic [IN_INDEX_W-1:0]   weight_write_input,
    input  logic signed [7:0]       weight_write_data,

    // ---------------- host: load biases ----------------
    input  logic                    bias_write_enable,
    input  logic [OUT_INDEX_W-1:0]  bias_write_neuron,
    input  logic signed [31:0]      bias_write_data,

    // ---------------- host: read outputs ----------------
    input  logic [OUT_INDEX_W-1:0]  output_read_address,
    output logic signed [7:0]       output_read_data,

    // ---------------- status + performance counters ----------------
    output logic                    busy,
    output logic                    done,
    output logic [31:0]             cycle_count,
    output logic [31:0]             mac_active_cycles,
    output logic [31:0]             mac_operations
);

    // ============================================================
    // Controller
    // ============================================================

    logic [OUT_DIM_W-1:0]  output_dim;
    logic [4:0]            output_shift;
    logic                  relu_enable;
    logic                  run_started;
    logic [IN_INDEX_W-1:0] read_input_index;
    logic [BATCH_W-1:0]    read_batch;
    logic                  mac_enable;
    logic                  mac_first;
    logic [BATCH_W-1:0]    mac_batch;
    logic                  write_enable;
    logic [BATCH_W-1:0]    write_batch;

    controller #(
        .NUM_MACS(NUM_MACS),
        .MAX_INPUTS(MAX_INPUTS),
        .MAX_OUTPUTS(MAX_OUTPUTS)
    ) controller_inst (
        .clk(clk),
        .reset(reset),
        .start(start),
        .config_input_dim(config_input_dim),
        .config_output_dim(config_output_dim),
        .config_output_shift(config_output_shift),
        .config_relu(config_relu),
        .output_dim(output_dim),
        .output_shift(output_shift),
        .relu_enable(relu_enable),
        .busy(busy),
        .done(done),
        .run_started(run_started),
        .read_input_index(read_input_index),
        .read_batch(read_batch),
        .mac_enable(mac_enable),
        .mac_first(mac_first),
        .mac_batch(mac_batch),
        .write_enable(write_enable),
        .write_batch(write_batch)
    );

    // ============================================================
    // Input memory: one INT8 value per input feature, shared by all lanes
    // ============================================================

    logic [7:0] input_value;

    sync_ram #(.WIDTH(8), .DEPTH(MAX_INPUTS)) input_memory (
        .clk(clk),
        .write_enable(input_write_enable),
        .write_address(input_write_address),
        .write_data(input_write_data),
        .read_address(read_input_index),
        .read_data(input_value)
    );

    // ============================================================
    // Where does a host-written neuron live?  lane = n % NUM_MACS, row = n / NUM_MACS
    // ============================================================

    function automatic logic [31:0] lane_of(input logic [OUT_INDEX_W-1:0] neuron);
        lane_of = 32'(neuron) % NUM_MACS;
    endfunction

    function automatic logic [BATCH_W-1:0] row_of(input logic [OUT_INDEX_W-1:0] neuron);
        row_of = BATCH_W'(32'(neuron) / NUM_MACS);
    endfunction

    // Weight bank address = row * MAX_INPUTS + input_index
    function automatic logic [BANK_ADDR_W-1:0] bank_address(
        input logic [BATCH_W-1:0]    row,
        input logic [IN_INDEX_W-1:0] input_index
    );
        bank_address = BANK_ADDR_W'(32'(row) * MAX_INPUTS + 32'(input_index));
    endfunction

    // ============================================================
    // MAC lanes (one per NUM_MACS)
    // ============================================================

    logic [NUM_MACS-1:0][7:0] lane_results;
    logic [NUM_MACS-1:0]      mac_lane_valid;    // lane holds a real neuron in stage 2
    logic [NUM_MACS-1:0]      write_lane_valid;  // lane holds a real neuron in stage 3

    genvar lane;
    generate
        for (lane = 0; lane < NUM_MACS; lane++) begin : lanes

            logic [7:0]         weight_value;
            logic [31:0]        bias_value;
            logic signed [31:0] accumulator;

            // This lane's slice of the weights
            sync_ram #(.WIDTH(8), .DEPTH(BANK_DEPTH)) weight_bank (
                .clk(clk),
                .write_enable(weight_write_enable && lane_of(weight_write_neuron) == lane),
                .write_address(bank_address(row_of(weight_write_neuron), weight_write_input)),
                .write_data(weight_write_data),
                .read_address(bank_address(read_batch, read_input_index)),
                .read_data(weight_value)
            );

            // This lane's biases: one per row (batch)
            sync_ram #(.WIDTH(32), .DEPTH(MAX_BATCHES)) bias_bank (
                .clk(clk),
                .write_enable(bias_write_enable && lane_of(bias_write_neuron) == lane),
                .write_address(row_of(bias_write_neuron)),
                .write_data(bias_write_data),
                .read_address(read_batch),
                .read_data(bias_value)
            );

            mac_unit mac (
                .clk(clk),
                .reset(reset),
                .enable(mac_enable),
                .first(mac_first),
                .input_value(input_value),
                .weight(weight_value),
                .bias(bias_value),
                .accumulator(accumulator)
            );

            post_process post (
                .accumulator(accumulator),
                .output_shift(output_shift),
                .relu_enable(relu_enable),
                .result(lane_results[lane])
            );

            // Neuron number = batch * NUM_MACS + lane. In the last batch some
            // lanes may be past output_dim; they compute garbage that is never
            // written and never counted.
            assign mac_lane_valid[lane]   = (32'(mac_batch)   * NUM_MACS + lane) < 32'(output_dim);
            assign write_lane_valid[lane] = (32'(write_batch) * NUM_MACS + lane) < 32'(output_dim);
        end
    endgenerate

    // ============================================================
    // Output memory
    // ============================================================

    output_memory #(
        .NUM_MACS(NUM_MACS),
        .MAX_OUTPUTS(MAX_OUTPUTS)
    ) output_memory_inst (
        .clk(clk),
        .write_enable(write_enable),
        .write_base(OUT_INDEX_W'(32'(write_batch) * NUM_MACS)),
        .lane_valid(write_lane_valid),
        .write_data(lane_results),
        .read_address(output_read_address),
        .read_data(output_read_data)
    );

    // ============================================================
    // Performance counters
    // ============================================================

    perf_counters #(.NUM_MACS(NUM_MACS)) counters (
        .clk(clk),
        .reset(reset),
        .run_started(run_started),
        .busy(busy),
        .mac_enable(mac_enable),
        .mac_lane_valid(mac_lane_valid),
        .cycle_count(cycle_count),
        .mac_active_cycles(mac_active_cycles),
        .mac_operations(mac_operations)
    );

endmodule
