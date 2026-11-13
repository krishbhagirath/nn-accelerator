// INT8 fully-connected neural-network accelerator: runs a WHOLE network
// (up to MAX_LAYERS layers) from one start pulse.
//
// How the host (cocotb testbench or C++ runtime) uses it:
//
//   once per model:     write the layer table, all layers' weights and biases
//   once per inference: write the input features -> start -> wait done -> read outputs
//
// Every layer's weights stay resident on chip, and layer outputs feed the next
// layer through two ping-pong activation buffers, so the host only touches the
// network's first inputs and last outputs.
//
//                         layer table (dims, shift, relu per layer)
//                                   |
//                              controller --- perf counters
//                                   |
//   activation buffer A/B --input[i]--+--> lane 0: weight bank 0, bias bank 0, MAC, post-process --+
//   (current inputs)                  +--> lane 1: ...                                             +--> activation buffer B/A
//                                     +--> lane NUM_MACS-1: ...                                    +    (next inputs)
//
// NUM_MACS (compile time) = how many neurons are computed in parallel.
//
// Weight storage: neuron n of layer L lives in lane (n % NUM_MACS), row (n / NUM_MACS).
// Each lane has its own weight bank. Every layer gets a fixed slot in it:
//     bank address = (L * MAX_BATCHES + row) * MAX_INPUTS + input_index
// During compute every lane reads the SAME address, so one address counter
// feeds all lanes and each bank needs only one read port.
// NUM_MACS must be a power of two (1, 2, 4, 8, ...).

`timescale 1ns/1ps

module nn_accelerator_top #(
    parameter NUM_MACS    = 4,
    parameter MAX_INPUTS  = 64,
    parameter MAX_OUTPUTS = 32,
    parameter MAX_LAYERS  = 4,

    localparam IN_DIM_W     = $clog2(MAX_INPUTS + 1),   // holds 0..MAX_INPUTS
    localparam IN_INDEX_W   = $clog2(MAX_INPUTS),       // holds 0..MAX_INPUTS-1
    localparam OUT_DIM_W    = $clog2(MAX_OUTPUTS + 1),
    localparam OUT_INDEX_W  = $clog2(MAX_OUTPUTS),
    localparam LAYER_W      = (MAX_LAYERS > 1) ? $clog2(MAX_LAYERS) : 1,
    localparam NUM_LAYERS_W = $clog2(MAX_LAYERS + 1),
    localparam MAX_BATCHES  = (MAX_OUTPUTS + NUM_MACS - 1) / NUM_MACS,
    localparam BATCH_W      = (MAX_BATCHES > 1) ? $clog2(MAX_BATCHES) : 1,
    localparam ROWS         = MAX_LAYERS * MAX_BATCHES,  // neuron rows per lane, all layers
    localparam ROW_W        = $clog2(ROWS),
    localparam BANK_DEPTH   = ROWS * MAX_INPUTS,
    localparam BANK_ADDR_W  = $clog2(BANK_DEPTH),
    localparam BUFFER_ROWS  = MAX_INPUTS / NUM_MACS,       // activation buffer rows
    localparam BUFFER_ROW_W = (BUFFER_ROWS > 1) ? $clog2(BUFFER_ROWS) : 1
)(
    input  logic                    clk,
    input  logic                    reset,
    input  logic                    start,
    input  logic [NUM_LAYERS_W-1:0] config_num_layers,  // 1..MAX_LAYERS

    // ---------------- host: write the layer table ----------------
    input  logic                    layer_write_enable,
    input  logic [LAYER_W-1:0]      layer_write_index,
    input  logic [IN_DIM_W-1:0]     layer_write_input_dim,
    input  logic [OUT_DIM_W-1:0]    layer_write_output_dim,
    input  logic [4:0]              layer_write_output_shift,
    input  logic                    layer_write_relu,

    // ---------------- host: load weights ----------------
    // weight[layer][neuron][input] = data
    input  logic                    weight_write_enable,
    input  logic [LAYER_W-1:0]      weight_write_layer,
    input  logic [OUT_INDEX_W-1:0]  weight_write_neuron,
    input  logic [IN_INDEX_W-1:0]   weight_write_input,
    input  logic signed [7:0]       weight_write_data,

    // ---------------- host: load biases ----------------
    input  logic                    bias_write_enable,
    input  logic [LAYER_W-1:0]      bias_write_layer,
    input  logic [OUT_INDEX_W-1:0]  bias_write_neuron,
    input  logic signed [31:0]      bias_write_data,

    // ---------------- host: write the network's input features ----------------
    input  logic                    input_write_enable,
    input  logic [IN_INDEX_W-1:0]   input_write_address,
    input  logic signed [7:0]       input_write_data,

    // ---------------- host: read the network's outputs ----------------
    input  logic [IN_INDEX_W-1:0]   output_read_address,
    output logic signed [7:0]       output_read_data,

    // ---------------- status + performance counters ----------------
    output logic                    busy,
    output logic                    done,
    output logic [31:0]             cycle_count,
    output logic [31:0]             mac_active_cycles,
    output logic [31:0]             mac_operations
);

    // ============================================================
    // Layer table + controller
    // ============================================================

    logic [LAYER_W-1:0]    table_index;
    logic [IN_DIM_W-1:0]   table_input_dim;
    logic [OUT_DIM_W-1:0]  table_output_dim;
    logic [4:0]            table_output_shift;
    logic                  table_relu;

    layer_table #(
        .MAX_LAYERS(MAX_LAYERS),
        .MAX_INPUTS(MAX_INPUTS),
        .MAX_OUTPUTS(MAX_OUTPUTS)
    ) layer_table_inst (
        .clk(clk),
        .write_enable(layer_write_enable),
        .write_index(layer_write_index),
        .write_input_dim(layer_write_input_dim),
        .write_output_dim(layer_write_output_dim),
        .write_output_shift(layer_write_output_shift),
        .write_relu(layer_write_relu),
        .read_index(table_index),
        .read_input_dim(table_input_dim),
        .read_output_dim(table_output_dim),
        .read_output_shift(table_output_shift),
        .read_relu(table_relu)
    );

    logic [LAYER_W-1:0]    layer;
    logic [OUT_DIM_W-1:0]  output_dim;
    logic [4:0]            output_shift;
    logic                  relu_enable;
    logic                  buffer_select;
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
        .MAX_OUTPUTS(MAX_OUTPUTS),
        .MAX_LAYERS(MAX_LAYERS)
    ) controller_inst (
        .clk(clk),
        .reset(reset),
        .start(start),
        .config_num_layers(config_num_layers),
        .table_index(table_index),
        .table_input_dim(table_input_dim),
        .table_output_dim(table_output_dim),
        .table_output_shift(table_output_shift),
        .table_relu(table_relu),
        .layer(layer),
        .output_dim(output_dim),
        .output_shift(output_shift),
        .relu_enable(relu_enable),
        .buffer_select(buffer_select),
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
    // Where does a neuron live?  lane = n % NUM_MACS, row = n / NUM_MACS
    // ============================================================

    function automatic logic [31:0] lane_of(input logic [OUT_INDEX_W-1:0] neuron);
        lane_of = 32'(neuron) % NUM_MACS;
    endfunction

    // Row inside the lane's bank, counting every layer's slot:  L * MAX_BATCHES + batch
    function automatic logic [ROW_W-1:0] row_of(input logic [LAYER_W-1:0] layer_index,
                                                input logic [BATCH_W-1:0] batch);
        row_of = ROW_W'(32'(layer_index) * MAX_BATCHES + 32'(batch));
    endfunction

    function automatic logic [BATCH_W-1:0] batch_of(input logic [OUT_INDEX_W-1:0] neuron);
        batch_of = BATCH_W'(32'(neuron) / NUM_MACS);
    endfunction

    // Weight bank address = row * MAX_INPUTS + input_index
    function automatic logic [BANK_ADDR_W-1:0] bank_address(
        input logic [ROW_W-1:0]      row,
        input logic [IN_INDEX_W-1:0] input_index
    );
        bank_address = BANK_ADDR_W'(32'(row) * MAX_INPUTS + 32'(input_index));
    endfunction

    logic [ROW_W-1:0] read_row;
    assign read_row = row_of(layer, read_batch);

    // ============================================================
    // Ping-pong activation buffers
    //   buffer[buffer_select]  : the current layer's inputs
    //   buffer[!buffer_select] : receives the current layer's outputs
    // The host always talks to buffer[buffer_select]: before start it holds
    // the inputs, after done (buffer_select has flipped) it holds the outputs.
    // ============================================================

    logic [NUM_MACS-1:0][7:0] lane_results;
    logic [NUM_MACS-1:0]      mac_lane_valid;    // lane holds a real neuron in stage 2
    logic [NUM_MACS-1:0]      write_lane_valid;  // lane holds a real neuron in stage 3

    logic [7:0] buffer_read_data [2];
    logic [7:0] buffer_host_data [2];
    logic [7:0] input_value;

    genvar b;
    generate
        for (b = 0; b < 2; b++) begin : buffers
            activation_buffer #(
                .NUM_MACS(NUM_MACS),
                .DEPTH(MAX_INPUTS)
            ) buffer (
                .clk(clk),
                .host_write_enable(input_write_enable && buffer_select == b),
                .host_write_address(input_write_address),
                .host_write_data(input_write_data),
                .row_write_enable(write_enable && buffer_select != b),
                .row_write_index(BUFFER_ROW_W'(write_batch)),   // batch b fills row b
                .lane_valid(write_lane_valid),
                .row_write_data(lane_results),
                .read_address(read_input_index),
                .read_data(buffer_read_data[b]),
                .host_read_address(output_read_address),
                .host_read_data(buffer_host_data[b])
            );
        end
    endgenerate

    assign input_value      = buffer_read_data[buffer_select];
    assign output_read_data = buffer_host_data[buffer_select];

    // ============================================================
    // MAC lanes (one per NUM_MACS)
    // ============================================================

    genvar lane;
    generate
        for (lane = 0; lane < NUM_MACS; lane++) begin : lanes

            logic [7:0]         weight_value;
            logic [31:0]        bias_value;
            logic signed [31:0] accumulator;

            // This lane's slice of every layer's weights
            sync_ram #(.WIDTH(8), .DEPTH(BANK_DEPTH)) weight_bank (
                .clk(clk),
                .write_enable(weight_write_enable && lane_of(weight_write_neuron) == lane),
                .write_address(bank_address(row_of(weight_write_layer, batch_of(weight_write_neuron)),
                                            weight_write_input)),
                .write_data(weight_write_data),
                .read_address(bank_address(read_row, read_input_index)),
                .read_data(weight_value)
            );

            // This lane's biases: one per row
            sync_ram #(.WIDTH(32), .DEPTH(ROWS)) bias_bank (
                .clk(clk),
                .write_enable(bias_write_enable && lane_of(bias_write_neuron) == lane),
                .write_address(row_of(bias_write_layer, batch_of(bias_write_neuron))),
                .write_data(bias_write_data),
                .read_address(read_row),
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
    // Performance counters (cover the whole network run)
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
