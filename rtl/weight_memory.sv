// Stores the neural network's 128 signed 8-bit weights.
// Loads one weight per clock cycle and reads 4 weights in parallel.
// The 4 read outputs will feed the 4-MAC array during inference.

module weight_memory (
    input  logic clk,

    // Write interface - used to load weights into memory
    input  logic write_enable,
    input  logic [6:0] write_address,
    input  logic signed [7:0] write_data,

    // Four read addresses allow 4 weights to be accessed in parallel
    input  logic [6:0] read_address0,
    input  logic [6:0] read_address1,
    input  logic [6:0] read_address2,
    input  logic [6:0] read_address3,

    // Weight values sent to the 4 MACs
    output logic signed [7:0] read_data0,
    output logic signed [7:0] read_data1,
    output logic signed [7:0] read_data2,
    output logic signed [7:0] read_data3
);

// 128 memory locations, each storing one signed 8-bit weight
logic signed [7:0] memory [0:127];

// Write one weight on each rising clock edge when enabled
always_ff @(posedge clk) begin
    if (write_enable)
        memory[write_address] <= write_data;
end

// Read four weights in parallel
always_comb begin
    read_data0 = memory[read_address0];
    read_data1 = memory[read_address1];
    read_data2 = memory[read_address2];
    read_data3 = memory[read_address3];
end

endmodule