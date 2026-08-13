module input_memory (
    input  logic clk,

    input  logic write_enable,
    input  logic [2:0] write_address,
    input  logic signed [7:0] write_data,

    input  logic [2:0] read_address,
    output logic signed [7:0] read_data
);

logic signed [7:0] memory [0:7];

always_ff @(posedge clk) begin
    if (write_enable)
        memory[write_address] <= write_data;
end

always_comb begin
    read_data = memory[read_address];
end

endmodule