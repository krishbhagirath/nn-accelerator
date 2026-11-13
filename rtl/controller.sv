// Sequences ONE fully-connected layer through the MAC lanes.
//
// Work is split into "batches": batch b computes neurons
//     b*NUM_MACS ... b*NUM_MACS + NUM_MACS-1     (one neuron per lane)
// and needs  input_dim  cycles, one per input feature. There are
//     num_batches = ceil(output_dim / NUM_MACS)
// batches, and the last one may be partial (extra lanes are ignored).
//
// The datapath is a 3-stage pipeline, because the memories take one cycle to
// return data:
//
//   stage 1 (READ)  : controller presents (batch, input_index) to the memories
//   stage 2 (MAC)   : data arrives, every lane does acc += input * weight
//   stage 3 (WRITE) : after a neuron's last input, results go to output memory
//
// The control flags (mac_enable, mac_first, write_enable...) are simply the
// stage-1 conditions delayed by one or two clock cycles, so they line up with
// the data they describe. Because of the pipeline, batch b+1 starts reading
// while batch b is still finishing, so there are no idle cycles between
// batches. Total busy time for one layer:
//
//     num_batches * input_dim  +  2   (2 = pipeline drain at the very end)

`timescale 1ns/1ps

module controller #(
    parameter NUM_MACS    = 4,
    parameter MAX_INPUTS  = 64,
    parameter MAX_OUTPUTS = 32,
    localparam IN_DIM_W   = $clog2(MAX_INPUTS + 1),
    localparam IN_INDEX_W = $clog2(MAX_INPUTS),
    localparam OUT_DIM_W  = $clog2(MAX_OUTPUTS + 1),
    localparam MAX_BATCHES = (MAX_OUTPUTS + NUM_MACS - 1) / NUM_MACS,
    localparam BATCH_W    = (MAX_BATCHES > 1) ? $clog2(MAX_BATCHES) : 1
)(
    input  logic                  clk,
    input  logic                  reset,
    input  logic                  start,

    // Layer configuration from the host. Captured when start is accepted, so
    // the host may change these while the layer runs.
    input  logic [IN_DIM_W-1:0]   config_input_dim,
    input  logic [OUT_DIM_W-1:0]  config_output_dim,
    input  logic [4:0]            config_output_shift,
    input  logic                  config_relu,

    // The captured configuration, used by the rest of the datapath
    output logic [OUT_DIM_W-1:0]  output_dim,
    output logic [4:0]            output_shift,
    output logic                  relu_enable,

    output logic                  busy,
    output logic                  done,
    output logic                  run_started,   // 1-cycle pulse: start accepted

    // Stage 1: memory read addresses
    output logic [IN_INDEX_W-1:0] read_input_index,
    output logic [BATCH_W-1:0]    read_batch,

    // Stage 2: MAC control (lines up with memory read data)
    output logic                  mac_enable,
    output logic                  mac_first,
    output logic [BATCH_W-1:0]    mac_batch,

    // Stage 3: write finished outputs
    output logic                  write_enable,
    output logic [BATCH_W-1:0]    write_batch
);

    typedef enum logic [1:0] {
        IDLE,     // waiting for start
        COMPUTE,  // stage 1 is issuing reads
        DRAIN,    // no more reads; waiting for the last batch to be written
        DONE      // results ready; wait for host to drop start
    } state_t;

    state_t state;

    logic [IN_DIM_W-1:0] input_dim;
    logic [BATCH_W-1:0]  last_batch;
    logic                mac_last;

    // ceil(output_dim / NUM_MACS) - 1, computed once per layer from the config.
    function automatic logic [BATCH_W-1:0] last_batch_index(input logic [OUT_DIM_W-1:0] dim);
        last_batch_index = BATCH_W'((32'(dim) + NUM_MACS - 1) / NUM_MACS - 1);
    endfunction

    logic issuing;
    logic last_input;
    assign issuing    = (state == COMPUTE);
    assign last_input = (read_input_index == IN_INDEX_W'(input_dim - 1));

    assign busy        = (state == COMPUTE) || (state == DRAIN);
    assign done        = (state == DONE);
    assign run_started = (state == IDLE) && start;

    // --------------------------------------------------------------
    // Stage 1: walk through (batch, input_index), one pair per cycle
    // --------------------------------------------------------------
    always_ff @(posedge clk) begin
        if (reset) begin
            state            <= IDLE;
            read_input_index <= 0;
            read_batch       <= 0;
        end else begin
            case (state)
                IDLE: begin
                    if (start) begin
                        input_dim        <= config_input_dim;
                        output_dim       <= config_output_dim;
                        output_shift     <= config_output_shift;
                        relu_enable      <= config_relu;
                        last_batch       <= last_batch_index(config_output_dim);
                        read_input_index <= 0;
                        read_batch       <= 0;
                        state            <= COMPUTE;
                    end
                end

                COMPUTE: begin
                    if (!last_input) begin
                        read_input_index <= read_input_index + 1;
                    end else begin
                        // Finished all inputs for this batch of neurons
                        read_input_index <= 0;
                        if (read_batch == last_batch)
                            state <= DRAIN;
                        else
                            read_batch <= read_batch + 1;
                    end
                end

                DRAIN: begin
                    // Leave once the final batch has been written (stage 3)
                    if (write_enable && write_batch == last_batch)
                        state <= DONE;
                end

                DONE: begin
                    if (!start)
                        state <= IDLE;
                end
            endcase
        end
    end

    // --------------------------------------------------------------
    // Stages 2 and 3: the same control information, delayed so it
    // arrives together with the data it describes
    // --------------------------------------------------------------
    always_ff @(posedge clk) begin
        if (reset) begin
            mac_enable   <= 0;
            mac_first    <= 0;
            mac_last     <= 0;
            mac_batch    <= 0;
            write_enable <= 0;
            write_batch  <= 0;
        end else begin
            mac_enable   <= issuing;
            mac_first    <= issuing && (read_input_index == 0);
            mac_last     <= issuing && last_input;
            mac_batch    <= read_batch;

            // A neuron is finished one cycle after its last MAC
            write_enable <= mac_enable && mac_last;
            write_batch  <= mac_batch;
        end
    end

endmodule
