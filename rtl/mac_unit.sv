// build one mac unit

module mac_unit (
    input  logic clk, // logic: 0 or 1
    input  logic reset,
    input  logic clear,
    input  logic enable,

    input  logic signed [7:0] input_value,
    input  logic signed [7:0] weight,

    output logic signed [31:0] accumulator // 32-bit signed accumulator, larger output than input
);

always_ff @(posedge clk) begin // only update on rising edge of clk
    if (reset)
        accumulator <= 0; // <= is non-blocking, everything occurs at clock edge (not sequentially)
    else if (clear)
        accumulator <= 0;
    else if (enable)
        accumulator <= accumulator + (input_value * weight);
end

endmodule

