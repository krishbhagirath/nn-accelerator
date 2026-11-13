// Simple single-port-write, single-port-read memory with a ONE-CYCLE read latency.
//
// Why synchronous read? Real on-chip SRAM / FPGA block RAM only works this way:
// you give it an address on one clock edge and the data appears after that edge.
// Writing the RTL like this lets synthesis map the memory to real RAM blocks
// instead of thousands of flip-flops.
//
// The data is stored as raw bits; the user decides whether it is signed.

`timescale 1ns/1ps

module sync_ram #(
    parameter WIDTH = 8,
    parameter DEPTH = 64,
    localparam ADDR_W = (DEPTH > 1) ? $clog2(DEPTH) : 1
)(
    input  logic              clk,

    input  logic              write_enable,
    input  logic [ADDR_W-1:0] write_address,
    input  logic [WIDTH-1:0]  write_data,

    input  logic [ADDR_W-1:0] read_address,
    output logic [WIDTH-1:0]  read_data      // valid one cycle after read_address
);

    logic [WIDTH-1:0] memory [DEPTH];

    always_ff @(posedge clk) begin
        if (write_enable)
            memory[write_address] <= write_data;

        read_data <= memory[read_address];
    end

endmodule
