// Holds one vector of INT8 activations (a layer's inputs or outputs).
//
// The top level has TWO of these and swaps their roles after every layer
// ("ping-pong" buffering):
//
//     layer 0:  reads buffer A, writes buffer B
//     layer 1:  reads buffer B, writes buffer A
//     layer 2:  reads buffer A, writes buffer B ...
//
// so a layer's outputs are already in place as the next layer's inputs, and
// nothing has to be copied.
//
// Storage is organized in ROWS of NUM_MACS values, because that is how the MAC
// lanes produce results: batch b writes neurons b*NUM_MACS ... b*NUM_MACS+NUM_MACS-1,
// i.e. exactly row b, with lane k going to column k. Element i lives at
//     row = i / NUM_MACS,  column = i % NUM_MACS
// (NUM_MACS is a power of two, so these are just bit slices.)
//
// Ports:
//   host write    one value per cycle (the network's input features)
//   row write     a whole row at once, one value per MAC lane
//                 (lanes past output_dim are switched off with lane_valid)
//   compute read  ONE-cycle latency, like the weight memories, so the
//                 activation and its weight arrive in the same cycle
//   host read     combinational (the network's final outputs)
//
// It is a small register array (DEPTH x 8 bits). It can't be block RAM because
// NUM_MACS lanes write to it in the same cycle.

`timescale 1ns/1ps

module activation_buffer #(
    parameter NUM_MACS = 4,
    parameter DEPTH    = 64,
    localparam INDEX_W = $clog2(DEPTH),
    localparam NUM_ROWS = DEPTH / NUM_MACS,
    localparam ROW_W   = (NUM_ROWS > 1) ? $clog2(NUM_ROWS) : 1
)(
    input  logic                     clk,

    input  logic                     host_write_enable,
    input  logic [INDEX_W-1:0]       host_write_address,
    input  logic [7:0]               host_write_data,

    input  logic                     row_write_enable,
    input  logic [ROW_W-1:0]         row_write_index,
    input  logic [NUM_MACS-1:0]      lane_valid,
    input  logic [NUM_MACS-1:0][7:0] row_write_data,

    input  logic [INDEX_W-1:0]       read_address,
    output logic [7:0]               read_data,       // valid one cycle after read_address

    input  logic [INDEX_W-1:0]       host_read_address,
    output logic [7:0]               host_read_data   // same cycle
);

    logic [NUM_MACS-1:0][7:0] rows [NUM_ROWS];

    function automatic logic [ROW_W-1:0] row_of(input logic [INDEX_W-1:0] index);
        row_of = ROW_W'(32'(index) / NUM_MACS);
    endfunction

    function automatic logic [31:0] column_of(input logic [INDEX_W-1:0] index);
        column_of = 32'(index) % NUM_MACS;
    endfunction

    always_ff @(posedge clk) begin
        if (host_write_enable)
            rows[row_of(host_write_address)][column_of(host_write_address)] <= host_write_data;

        if (row_write_enable) begin
            for (int lane = 0; lane < NUM_MACS; lane++) begin
                if (lane_valid[lane])
                    rows[row_write_index][lane] <= row_write_data[lane];
            end
        end

        read_data <= rows[row_of(read_address)][column_of(read_address)];
    end

    assign host_read_data = rows[row_of(host_read_address)][column_of(host_read_address)];

endmodule
