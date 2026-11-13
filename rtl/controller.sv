// Runs a WHOLE network: every layer, back to back, from one start pulse.
//
// ---------------------------------------------------------------------------
// Inside one layer
// ---------------------------------------------------------------------------
// Work is split into "batches": batch b computes neurons
//     b*NUM_MACS ... b*NUM_MACS + NUM_MACS-1     (one neuron per lane)
// and needs  input_dim  cycles, one per input feature. There are
//     num_batches = ceil(output_dim / NUM_MACS)
// batches, and the last one may be partial (extra lanes are ignored).
//
// The datapath is a 3-stage pipeline, because the memories take one cycle to
// return data:
//
//   stage 1 (READ)  : controller presents (layer, batch, input_index) to the memories
//   stage 2 (MAC)   : data arrives, every lane does acc += input * weight
//   stage 3 (WRITE) : after a neuron's last input, results go to the output buffer
//
// The control flags (mac_enable, mac_first, write_enable...) are simply the
// stage-1 conditions delayed by one or two clock cycles, so they line up with
// the data they describe. Because of the pipeline, batch b+1 starts reading
// while batch b is still finishing, so there are no idle cycles between batches.
//
// ---------------------------------------------------------------------------
// Between layers
// ---------------------------------------------------------------------------
// After a layer's last read, the controller waits 2 cycles (DRAIN) for the
// pipeline to write the last batch. On the cycle that final write happens it:
//   - flips  buffer_select  so the outputs just written become the next inputs
//   - loads the next layer's entry from the layer table
//   - goes straight back to COMPUTE (or to DONE after the last layer)
// So the busy time of a whole network is
//
//     sum over layers of ( num_batches * input_dim + 2 )
//
// ---------------------------------------------------------------------------
// States
// ---------------------------------------------------------------------------
//   IDLE    -> start         -> COMPUTE (layer 0)
//   COMPUTE -> last read      -> DRAIN
//   DRAIN   -> last write     -> COMPUTE (next layer)  or  DONE (no layers left)
//   DONE    -> start dropped  -> IDLE

`timescale 1ns/1ps

module controller #(
    parameter NUM_MACS    = 4,
    parameter MAX_INPUTS  = 64,
    parameter MAX_OUTPUTS = 32,
    parameter MAX_LAYERS  = 4,
    localparam IN_DIM_W    = $clog2(MAX_INPUTS + 1),
    localparam IN_INDEX_W  = $clog2(MAX_INPUTS),
    localparam OUT_DIM_W   = $clog2(MAX_OUTPUTS + 1),
    localparam MAX_BATCHES = (MAX_OUTPUTS + NUM_MACS - 1) / NUM_MACS,
    localparam BATCH_W     = (MAX_BATCHES > 1) ? $clog2(MAX_BATCHES) : 1,
    localparam LAYER_W     = (MAX_LAYERS > 1) ? $clog2(MAX_LAYERS) : 1,
    localparam NUM_LAYERS_W = $clog2(MAX_LAYERS + 1)
)(
    input  logic                    clk,
    input  logic                    reset,
    input  logic                    start,
    input  logic [NUM_LAYERS_W-1:0] config_num_layers,  // 1..MAX_LAYERS, captured at start

    // Layer table lookup: the controller asks for an entry, the table answers
    // in the same cycle.
    output logic [LAYER_W-1:0]      table_index,
    input  logic [IN_DIM_W-1:0]     table_input_dim,
    input  logic [OUT_DIM_W-1:0]    table_output_dim,
    input  logic [4:0]              table_output_shift,
    input  logic                    table_relu,

    // The layer being run, used by the rest of the datapath
    output logic [LAYER_W-1:0]      layer,
    output logic [OUT_DIM_W-1:0]    output_dim,
    output logic [4:0]              output_shift,
    output logic                    relu_enable,

    // Which activation buffer holds the current layer's inputs (0 or 1).
    // The other buffer receives its outputs. Flips after every layer, so
    // after done it points at the network's final outputs.
    output logic                    buffer_select,

    output logic                    busy,
    output logic                    done,
    output logic                    run_started,   // 1-cycle pulse: start accepted

    // Stage 1: memory read addresses
    output logic [IN_INDEX_W-1:0]   read_input_index,
    output logic [BATCH_W-1:0]      read_batch,

    // Stage 2: MAC control (lines up with memory read data)
    output logic                    mac_enable,
    output logic                    mac_first,
    output logic [BATCH_W-1:0]      mac_batch,

    // Stage 3: write finished outputs
    output logic                    write_enable,
    output logic [BATCH_W-1:0]      write_batch
);

    typedef enum logic [1:0] {
        IDLE,     // waiting for start
        COMPUTE,  // stage 1 is issuing reads for the current layer
        DRAIN,    // no more reads; waiting for the layer's last batch to be written
        DONE      // network finished; wait for host to drop start
    } state_t;

    state_t state;

    logic [IN_DIM_W-1:0] input_dim;
    logic [BATCH_W-1:0]  last_batch;
    logic [LAYER_W-1:0]  last_layer;
    logic                mac_last;

    // ceil(output_dim / NUM_MACS) - 1
    function automatic logic [BATCH_W-1:0] last_batch_index(input logic [OUT_DIM_W-1:0] dim);
        last_batch_index = BATCH_W'((32'(dim) + NUM_MACS - 1) / NUM_MACS - 1);
    endfunction

    logic issuing;
    logic last_input;
    logic layer_finished;
    assign issuing        = (state == COMPUTE);
    assign last_input     = (read_input_index == IN_INDEX_W'(input_dim - 1));
    assign layer_finished = (state == DRAIN) && write_enable && (write_batch == last_batch);

    assign busy        = (state == COMPUTE) || (state == DRAIN);
    assign done        = (state == DONE);
    assign run_started = (state == IDLE) && start;

    // A new layer begins on start, or when a layer finishes and more are left.
    // table_index is the entry that layer needs: 0 when starting, else layer + 1.
    logic begin_layer;
    assign begin_layer = run_started || (layer_finished && layer != last_layer);
    assign table_index = (state == IDLE) ? '0 : layer + 1'b1;

    // --------------------------------------------------------------
    // Stage 1: walk through (layer, batch, input_index), one per cycle
    // --------------------------------------------------------------
    always_ff @(posedge clk) begin
        if (reset) begin
            state            <= IDLE;
            layer            <= 0;
            buffer_select    <= 0;
            read_input_index <= 0;
            read_batch       <= 0;
        end else begin
            case (state)
                IDLE: begin
                    if (start)
                        last_layer <= LAYER_W'(config_num_layers - 1);
                    // (moving to COMPUTE is handled by begin_layer below)
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
                    if (layer_finished) begin
                        // This layer's outputs become the next layer's inputs
                        buffer_select <= !buffer_select;
                        if (layer == last_layer)
                            state <= DONE;
                        // (otherwise begin_layer below starts the next layer)
                    end
                end

                DONE: begin
                    if (!start)
                        state <= IDLE;
                end
            endcase

            // Load the next layer's table entry into the working registers
            if (begin_layer) begin
                layer            <= table_index;
                input_dim        <= table_input_dim;
                output_dim       <= table_output_dim;
                output_shift     <= table_output_shift;
                relu_enable      <= table_relu;
                last_batch       <= last_batch_index(table_output_dim);
                read_input_index <= 0;
                read_batch       <= 0;
                state            <= COMPUTE;
            end
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
