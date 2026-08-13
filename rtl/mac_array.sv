module mac_array (
    input  logic clk,
    input  logic reset,
    input  logic clear,
    input  logic enable,

    input  logic signed [7:0] input_value,

    input  logic signed [7:0] weight0,
    input  logic signed [7:0] weight1,
    input  logic signed [7:0] weight2,
    input  logic signed [7:0] weight3,

    output logic signed [31:0] acc0,
    output logic signed [31:0] acc1,
    output logic signed [31:0] acc2,
    output logic signed [31:0] acc3
);

mac_unit mac0 (
    .clk(clk),
    .reset(reset),
    .clear(clear),
    .enable(enable),
    .input_value(input_value),
    .weight(weight0),
    .accumulator(acc0)
);

mac_unit mac1 (
    .clk(clk),
    .reset(reset),
    .clear(clear),
    .enable(enable),
    .input_value(input_value),
    .weight(weight1),
    .accumulator(acc1)
);

mac_unit mac2 (
    .clk(clk),
    .reset(reset),
    .clear(clear),
    .enable(enable),
    .input_value(input_value),
    .weight(weight2),
    .accumulator(acc2)
);

mac_unit mac3 (
    .clk(clk),
    .reset(reset),
    .clear(clear),
    .enable(enable),
    .input_value(input_value),
    .weight(weight3),
    .accumulator(acc3)
);

endmodule