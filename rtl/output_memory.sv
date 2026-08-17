// Stores the 16 final INT8 neuron outputs.
// The accelerator writes 4 neuron results in parallel after each MAC batch.
// Results can later be read individually by the host/testbench.

module output_memory (
    input logic clk,

    // Write 4 outputs in parallel
    input logic write_enable,

    input logic [3:0] write_address0,
    input logic [3:0] write_address1,
    input logic [3:0] write_address2,
    input logic [3:0] write_address3,

    input logic signed [7:0] write_data0,
    input logic signed [7:0] write_data1,
    input logic signed [7:0] write_data2,
    input logic signed [7:0] write_data3,

    // Host/testbench reads one result at a time
    input logic [3:0] read_address,
    output logic signed [7:0] read_data
);

logic signed [7:0] memory [0:15];

always_ff @(posedge clk) begin
    if (write_enable) begin
        memory[write_address0] <= write_data0;
        memory[write_address1] <= write_data1;
        memory[write_address2] <= write_data2;
        memory[write_address3] <= write_data3;
    end
end

always_comb begin
    read_data = memory[read_address];
end

endmodule