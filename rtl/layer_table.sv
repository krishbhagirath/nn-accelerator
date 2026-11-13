// The network description: one small entry per layer.
//
// The host writes one entry per layer before running the network:
//     entry[layer] = { input_dim, output_dim, output_shift, relu }
// The controller reads the entry of the layer it is about to run.
//
// It is just a handful of registers (MAX_LAYERS entries of 19 bits), so the
// read is combinational: the value for read_index is available in the same cycle.

`timescale 1ns/1ps

module layer_table #(
    parameter MAX_LAYERS  = 4,
    parameter MAX_INPUTS  = 64,
    parameter MAX_OUTPUTS = 32,
    localparam LAYER_W    = (MAX_LAYERS > 1) ? $clog2(MAX_LAYERS) : 1,
    localparam IN_DIM_W   = $clog2(MAX_INPUTS + 1),
    localparam OUT_DIM_W  = $clog2(MAX_OUTPUTS + 1)
)(
    input  logic                 clk,

    // host write port
    input  logic                 write_enable,
    input  logic [LAYER_W-1:0]   write_index,
    input  logic [IN_DIM_W-1:0]  write_input_dim,
    input  logic [OUT_DIM_W-1:0] write_output_dim,
    input  logic [4:0]           write_output_shift,
    input  logic                 write_relu,

    // controller read port
    input  logic [LAYER_W-1:0]   read_index,
    output logic [IN_DIM_W-1:0]  read_input_dim,
    output logic [OUT_DIM_W-1:0] read_output_dim,
    output logic [4:0]           read_output_shift,
    output logic                 read_relu
);

    logic [IN_DIM_W-1:0]  input_dim    [MAX_LAYERS];
    logic [OUT_DIM_W-1:0] output_dim   [MAX_LAYERS];
    logic [4:0]           output_shift [MAX_LAYERS];
    logic                 relu         [MAX_LAYERS];

    always_ff @(posedge clk) begin
        if (write_enable) begin
            input_dim[write_index]    <= write_input_dim;
            output_dim[write_index]   <= write_output_dim;
            output_shift[write_index] <= write_output_shift;
            relu[write_index]         <= write_relu;
        end
    end

    assign read_input_dim    = input_dim[read_index];
    assign read_output_dim   = output_dim[read_index];
    assign read_output_shift = output_shift[read_index];
    assign read_relu         = relu[read_index];

endmodule
