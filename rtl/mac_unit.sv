// One multiply-accumulate lane: accumulator += input * weight.
//
// Each lane computes ONE output neuron at a time. On the first input of a
// neuron the accumulator is loaded with that neuron's bias instead of its old
// value, so there is no separate "clear" or "add bias" step:
//
//     first input:  accumulator = bias        + input[0] * weight[0]
//     later inputs: accumulator = accumulator + input[i] * weight[i]

`timescale 1ns/1ps

module mac_unit (
    input  logic               clk,
    input  logic               reset,
    input  logic               enable,   // a valid (input, weight) pair is present
    input  logic               first,    // this is input 0 of a new neuron

    input  logic signed [7:0]  input_value,
    input  logic signed [7:0]  weight,
    input  logic signed [31:0] bias,

    output logic signed [31:0] accumulator
);

    // INT8 x INT8 always fits in 16 bits: -128 * -128 = 16384.
    logic signed [15:0] product;
    assign product = input_value * weight;

    always_ff @(posedge clk) begin
        if (reset)
            accumulator <= 0;
        else if (enable)
            accumulator <= (first ? bias : accumulator) + 32'(product);
    end

endmodule
