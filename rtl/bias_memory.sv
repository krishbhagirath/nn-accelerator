// Stores one signed 32-bit bias value for each of the 16 neurons.
// Supports loading one bias per clock cycle and reading 4 biases in parallel.
// The 4 outputs correspond to the 4 neurons being processed by the MAC array.

module bias_memory (
    input  logic clk,

    // Write interface
    input  logic write_enable,
    input  logic [3:0] write_address,
    input  logic signed [31:0] write_data,

    // Four parallel read addresses
    input  logic [3:0] read_address0,
    input  logic [3:0] read_address1,
    input  logic [3:0] read_address2,
    input  logic [3:0] read_address3,

    // Four bias outputs
    output logic signed [31:0] read_data0,
    output logic signed [31:0] read_data1,
    output logic signed [31:0] read_data2,
    output logic signed [31:0] read_data3
);

logic signed [31:0] memory [0:15];

// Load one bias on a rising clock edge
always_ff @(posedge clk) begin
    if (write_enable)
        memory[write_address] <= write_data;
end

// Read four biases in parallel
always_comb begin
    read_data0 = memory[read_address0];
    read_data1 = memory[read_address1];
    read_data2 = memory[read_address2];
    read_data3 = memory[read_address3];
end

endmodule