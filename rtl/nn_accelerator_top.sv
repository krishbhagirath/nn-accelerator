// Top-level neural network accelerator.
// Connects the controller, memories, 4-MAC array, post-processing, and output memory.
// External write ports allow a host/testbench to load inputs, weights, and biases.

`timescale 1ns/1ps

module nn_accelerator_top #(
    parameter OUTPUT_SHIFT = 9
)(
    input logic clk,
    input logic reset,
    input logic start,

    // -----------------------------
    // Host interface: load inputs
    // -----------------------------
    input logic input_write_enable,
    input logic [2:0] input_write_address,
    input logic signed [7:0] input_write_data,

    // -----------------------------
    // Host interface: load weights
    // -----------------------------
    input logic weight_write_enable,
    input logic [6:0] weight_write_address,
    input logic signed [7:0] weight_write_data,

    // -----------------------------
    // Host interface: load biases
    // -----------------------------
    input logic bias_write_enable,
    input logic [3:0] bias_write_address,
    input logic signed [31:0] bias_write_data,

    // -----------------------------
    // Host interface: read outputs
    // -----------------------------
    input logic [3:0] output_read_address,
    output logic signed [7:0] output_read_data,

    // Accelerator status
    output logic busy,
    output logic done
);

    // ============================================================
    // Controller signals
    // ============================================================

    logic mac_clear;
    logic mac_enable;
    logic output_write_enable;

    logic [2:0] input_index;   // Which input: 0-7
    logic [1:0] batch_index;   // Which neuron batch: 0-3


    // ============================================================
    // Input memory → MAC array
    // ============================================================

    logic signed [7:0] input_value;


    // ============================================================
    // Weight memory → 4 MAC lanes
    // ============================================================

    logic [6:0] weight_address0;
    logic [6:0] weight_address1;
    logic [6:0] weight_address2;
    logic [6:0] weight_address3;

    logic signed [7:0] weight0;
    logic signed [7:0] weight1;
    logic signed [7:0] weight2;
    logic signed [7:0] weight3;


    // ============================================================
    // MAC array outputs
    // ============================================================

    logic signed [31:0] acc0;
    logic signed [31:0] acc1;
    logic signed [31:0] acc2;
    logic signed [31:0] acc3;


    // ============================================================
    // Bias memory
    // ============================================================

    logic [3:0] bias_address0;
    logic [3:0] bias_address1;
    logic [3:0] bias_address2;
    logic [3:0] bias_address3;

    logic signed [31:0] bias0;
    logic signed [31:0] bias1;
    logic signed [31:0] bias2;
    logic signed [31:0] bias3;


    // ============================================================
    // Post-processing outputs
    // ============================================================

    logic signed [7:0] out0;
    logic signed [7:0] out1;
    logic signed [7:0] out2;
    logic signed [7:0] out3;


    // ============================================================
    // Output memory addresses
    // ============================================================

    logic [3:0] output_address0;
    logic [3:0] output_address1;
    logic [3:0] output_address2;
    logic [3:0] output_address3;


    // ============================================================
    // Address generation
    //
    // Each batch contains 4 neurons.
    //
    // batch 0 → neurons 0-3
    // batch 1 → neurons 4-7
    // batch 2 → neurons 8-11
    // batch 3 → neurons 12-15
    // ============================================================

    always_comb begin

        // Each neuron has 8 weights.
        //
        // Example:
        // batch 1 = neurons 4,5,6,7
        //
        // neuron 4 weights begin at 4*8 = address 32
        // neuron 5 weights begin at 5*8 = address 40
        // etc.

        weight_address0 = {batch_index, 5'b00000}
                        + input_index;

        weight_address1 = {batch_index, 5'b00000}
                        + 7'd8
                        + input_index;

        weight_address2 = {batch_index, 5'b00000}
                        + 7'd16
                        + input_index;

        weight_address3 = {batch_index, 5'b00000}
                        + 7'd24
                        + input_index;


        // One bias per neuron
        bias_address0 = {batch_index, 2'b00};
        bias_address1 = {batch_index, 2'b00} + 4'd1;
        bias_address2 = {batch_index, 2'b00} + 4'd2;
        bias_address3 = {batch_index, 2'b00} + 4'd3;


        // Finished outputs go into the same neuron-number locations
        output_address0 = bias_address0;
        output_address1 = bias_address1;
        output_address2 = bias_address2;
        output_address3 = bias_address3;

    end


    // ============================================================
    // Controller
    // ============================================================

    controller controller_inst (
        .clk(clk),
        .reset(reset),
        .start(start),

        .busy(busy),
        .done(done),

        .mac_clear(mac_clear),
        .mac_enable(mac_enable),
        .output_write_enable(output_write_enable),

        .input_index(input_index),
        .batch_index(batch_index)
    );


    // ============================================================
    // Input Memory
    // ============================================================

    input_memory input_memory_inst (
        .clk(clk),

        .write_enable(input_write_enable),
        .write_address(input_write_address),
        .write_data(input_write_data),

        // Controller chooses which input is currently needed
        .read_address(input_index),
        .read_data(input_value)
    );


    // ============================================================
    // Weight Memory
    // ============================================================

    weight_memory weight_memory_inst (
        .clk(clk),

        .write_enable(weight_write_enable),
        .write_address(weight_write_address),
        .write_data(weight_write_data),

        .read_address0(weight_address0),
        .read_address1(weight_address1),
        .read_address2(weight_address2),
        .read_address3(weight_address3),

        .read_data0(weight0),
        .read_data1(weight1),
        .read_data2(weight2),
        .read_data3(weight3)
    );


    // ============================================================
    // 4-MAC Array
    // ============================================================

    mac_array mac_array_inst (
        .clk(clk),
        .reset(reset),

        .clear(mac_clear),
        .enable(mac_enable),

        // Same input is broadcast to all four MACs
        .input_value(input_value),

        // Each MAC gets its own neuron's weight
        .weight0(weight0),
        .weight1(weight1),
        .weight2(weight2),
        .weight3(weight3),

        .acc0(acc0),
        .acc1(acc1),
        .acc2(acc2),
        .acc3(acc3)
    );


    // ============================================================
    // Bias Memory
    // ============================================================

    bias_memory bias_memory_inst (
        .clk(clk),

        .write_enable(bias_write_enable),
        .write_address(bias_write_address),
        .write_data(bias_write_data),

        .read_address0(bias_address0),
        .read_address1(bias_address1),
        .read_address2(bias_address2),
        .read_address3(bias_address3),

        .read_data0(bias0),
        .read_data1(bias1),
        .read_data2(bias2),
        .read_data3(bias3)
    );


    // ============================================================
    // Post Processing
    //
    // MAC result
    //    ↓
    // + bias
    //    ↓
    // scaling
    //    ↓
    // ReLU
    //    ↓
    // clamp to 0-127
    // ============================================================

    post_process #(
        .OUTPUT_SHIFT(OUTPUT_SHIFT)
    ) post_process_inst (
        .acc0(acc0),
        .acc1(acc1),
        .acc2(acc2),
        .acc3(acc3),

        .bias0(bias0),
        .bias1(bias1),
        .bias2(bias2),
        .bias3(bias3),

        .out0(out0),
        .out1(out1),
        .out2(out2),
        .out3(out3)
    );


    // ============================================================
    // Output Memory
    // ============================================================

    output_memory output_memory_inst (
        .clk(clk),

        // Controller enables this after a batch finishes
        .write_enable(output_write_enable),

        .write_address0(output_address0),
        .write_address1(output_address1),
        .write_address2(output_address2),
        .write_address3(output_address3),

        .write_data0(out0),
        .write_data1(out1),
        .write_data2(out2),
        .write_data3(out3),

        // Host/testbench can inspect results after DONE
        .read_address(output_read_address),
        .read_data(output_read_data)
    );

endmodule