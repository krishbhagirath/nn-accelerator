module mac_array_tb;

logic clk = 0;
logic reset;
logic clear;
logic enable;

logic signed [7:0] input_value;

logic signed [7:0] weight0;
logic signed [7:0] weight1;
logic signed [7:0] weight2;
logic signed [7:0] weight3;

logic signed [31:0] acc0;
logic signed [31:0] acc1;
logic signed [31:0] acc2;
logic signed [31:0] acc3;


// Instantiate the MAC array
mac_array dut (
    .clk(clk),
    .reset(reset),
    .clear(clear),
    .enable(enable),

    .input_value(input_value),

    .weight0(weight0),
    .weight1(weight1),
    .weight2(weight2),
    .weight3(weight3),

    .acc0(acc0),
    .acc1(acc1),
    .acc2(acc2),
    .acc3(acc3)
);


// Generate clock
always #5 clk = ~clk;


initial begin

    // Initial values
    reset = 1;
    clear = 0;
    enable = 0;

    input_value = 0;

    weight0 = 0;
    weight1 = 0;
    weight2 = 0;
    weight3 = 0;

    #10;

    // Release reset and start computing
    reset = 0;
    enable = 1;


    // Cycle 1
    input_value = 2;

    weight0 = 1;
    weight1 = 2;
    weight2 = -1;
    weight3 = 3;

    #10;


    // Cycle 2
    input_value = 5;

    weight0 = 2;
    weight1 = 1;
    weight2 = 1;
    weight3 = 0;

    #10;


    // Cycle 3
    input_value = 1;

    weight0 = 3;
    weight1 = 0;
    weight2 = 2;
    weight3 = 1;

    #10;


    // Cycle 4
    input_value = 7;

    weight0 = 4;
    weight1 = 1;
    weight2 = 0;
    weight3 = 2;

    #10;


    // Print results
    $display("acc0 = %0d (expected 43)", acc0);
    $display("acc1 = %0d (expected 16)", acc1);
    $display("acc2 = %0d (expected 5)", acc2);
    $display("acc3 = %0d (expected 21)", acc3);

    $finish;

end

endmodule