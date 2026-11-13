// Holds the finished INT8 outputs of one layer.
//
// After each batch, all NUM_MACS lanes write their neuron's result at once:
//     lane k of batch b is neuron  b * NUM_MACS + k
// Lanes whose neuron number is >= output_dim (the unused lanes of the last,
// partial batch) are switched off with lane_valid.
//
// The host reads results back one at a time (combinational read, so the value
// for output_read_address is available in the same cycle).

`timescale 1ns/1ps

module output_memory #(
    parameter NUM_MACS    = 4,
    parameter MAX_OUTPUTS = 32,
    localparam INDEX_W    = $clog2(MAX_OUTPUTS)
)(
    input  logic clk,

    input  logic                       write_enable,
    input  logic [INDEX_W-1:0]         write_base,    // neuron number of lane 0
    input  logic [NUM_MACS-1:0]        lane_valid,
    input  logic [NUM_MACS-1:0][7:0]   write_data,

    input  logic [INDEX_W-1:0]         read_address,
    output logic signed [7:0]          read_data
);

    logic [7:0] memory [MAX_OUTPUTS];

    always_ff @(posedge clk) begin
        if (write_enable) begin
            for (int lane = 0; lane < NUM_MACS; lane++) begin
                if (lane_valid[lane])
                    memory[write_base + INDEX_W'(lane)] <= write_data[lane];
            end
        end
    end

    assign read_data = memory[read_address];

endmodule
