// simulator file

module mac_unit_tb;

logic clk = 0;
logic reset;
logic clear;
logic enable;

logic signed [7:0] input_value;
logic signed [7:0] weight;

logic signed [31:0] accumulator;

mac_unit dut (
    .clk(clk),
    .reset(reset),
    .clear(clear),
    .enable(enable),
    .input_value(input_value),
    .weight(weight),
    .accumulator(accumulator)
); // init a mac unit instance

always #5 clk = ~clk; // wait 5 time units, then invert clk forever

initial begin
    reset = 1;
    clear = 0;
    enable = 0;
    input_value = 0;
    weight = 0;

    #10;

    reset = 0;
    enable = 1;

    input_value = 2;
    weight = 1;
    #10;

    input_value = 5;
    weight = 2;
    #10;

    input_value = 1;
    weight = 3;
    #10;

    input_value = 7;
    weight = 4;
    #10;

// Our original test should equal 43
$display("Test 1 - accumulator = %0d (expected 43)", accumulator);

// Test CLEAR
clear = 1;
#10;
clear = 0;

$display("Test 2 - after clear = %0d (expected 0)", accumulator);

// Test signed multiplication
input_value = -2;
weight = 3;
#10;

$display("Test 3 - signed multiply = %0d (expected -6)", accumulator);

// Test ENABLE = 0
enable = 0;
input_value = 10;
weight = 10;
#10;

$display("Test 4 - disabled = %0d (expected -6)", accumulator);

$finish;
end

endmodule
