// Turns one 32-bit accumulator into one INT8 output (purely combinational).
//
//   1. Requantize: arithmetic right shift by output_shift. The compiler picks
//      the shift so the largest calibration value just fits in INT8.
//      >>> rounds toward -infinity; the Python golden model does the same.
//   2. Clamp:
//        hidden layer (relu_enable=1): 0 .. 127   (ReLU + saturation)
//        final layer  (relu_enable=0): -128 .. 127 (signed logits)

`timescale 1ns/1ps

module post_process (
    input  logic signed [31:0] accumulator,
    input  logic        [4:0]  output_shift,
    input  logic               relu_enable,
    output logic signed [7:0]  result
);

    logic signed [31:0] shifted;
    logic               too_high;
    logic               too_low;

    assign shifted  = accumulator >>> output_shift;
    assign too_high = shifted > 32'sd127;
    assign too_low  = relu_enable ? (shifted < 32'sd0) : (shifted < -32'sd128);

    assign result = too_high ? 8'sd127 :
                    too_low  ? (relu_enable ? 8'sd0 : -8'sd128) :
                               shifted[7:0];

endmodule
