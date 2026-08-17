// Post-processes the 4 neuron sums produced by the MAC array.
// Adds bias, rescales the result, applies ReLU, and clamps to INT8 range 0-127.
// Processes all 4 neuron outputs in parallel.

module post_process #(
    parameter OUTPUT_SHIFT = 0
)(
    input logic signed [31:0] acc0,
    input logic signed [31:0] acc1,
    input logic signed [31:0] acc2,
    input logic signed [31:0] acc3,

    input logic signed [31:0] bias0,
    input logic signed [31:0] bias1,
    input logic signed [31:0] bias2,
    input logic signed [31:0] bias3,

    output logic signed [7:0] out0,
    output logic signed [7:0] out1,
    output logic signed [7:0] out2,
    output logic signed [7:0] out3
);

logic signed [31:0] sum0;
logic signed [31:0] sum1;
logic signed [31:0] sum2;
logic signed [31:0] sum3;

logic signed [31:0] scaled0;
logic signed [31:0] scaled1;
logic signed [31:0] scaled2;
logic signed [31:0] scaled3;

always_comb begin

    // Add each neuron's bias
    sum0 = acc0 + bias0;
    sum1 = acc1 + bias1;
    sum2 = acc2 + bias2;
    sum3 = acc3 + bias3;

    // Rescale before converting to 8-bit output
    scaled0 = sum0 >>> OUTPUT_SHIFT;
    scaled1 = sum1 >>> OUTPUT_SHIFT;
    scaled2 = sum2 >>> OUTPUT_SHIFT;
    scaled3 = sum3 >>> OUTPUT_SHIFT;

    // ReLU + saturation
    if (scaled0 < 0)
        out0 = 0;
    else if (scaled0 > 127)
        out0 = 127;
    else
        out0 = scaled0[7:0];

    if (scaled1 < 0)
        out1 = 0;
    else if (scaled1 > 127)
        out1 = 127;
    else
        out1 = scaled1[7:0];

    if (scaled2 < 0)
        out2 = 0;
    else if (scaled2 > 127)
        out2 = 127;
    else
        out2 = scaled2[7:0];

    if (scaled3 < 0)
        out3 = 0;
    else if (scaled3 > 127)
        out3 = 127;
    else
        out3 = scaled3[7:0];

end

endmodule



// must still implement another method of scaling, besides clamping (clamping leads to loss of information)