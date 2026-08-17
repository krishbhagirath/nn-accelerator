module nn_accelerator_top_tb;

logic clk = 0;
logic reset;
logic start;

logic input_write_enable;
logic [2:0] input_write_address;
logic signed [7:0] input_write_data;

logic weight_write_enable;
logic [6:0] weight_write_address;
logic signed [7:0] weight_write_data;

logic bias_write_enable;
logic [3:0] bias_write_address;
logic signed [31:0] bias_write_data;

logic [3:0] output_read_address;
logic signed [7:0] output_read_data;

logic busy;
logic done;


// Instantiate complete accelerator
nn_accelerator_top dut (
    .clk(clk),
    .reset(reset),
    .start(start),

    .input_write_enable(input_write_enable),
    .input_write_address(input_write_address),
    .input_write_data(input_write_data),

    .weight_write_enable(weight_write_enable),
    .weight_write_address(weight_write_address),
    .weight_write_data(weight_write_data),

    .bias_write_enable(bias_write_enable),
    .bias_write_address(bias_write_address),
    .bias_write_data(bias_write_data),

    .output_read_address(output_read_address),
    .output_read_data(output_read_data),

    .busy(busy),
    .done(done)
);


// Clock
always #5 clk = ~clk;


// Helper task: write one input
task write_input(
    input [2:0] address,
    input signed [7:0] data
);
begin
    input_write_enable = 1;
    input_write_address = address;
    input_write_data = data;

    #10;

    input_write_enable = 0;
end
endtask


// Helper task: write one weight
task write_weight(
    input [6:0] address,
    input signed [7:0] data
);
begin
    weight_write_enable = 1;
    weight_write_address = address;
    weight_write_data = data;

    #10;

    weight_write_enable = 0;
end
endtask


// Helper task: write one bias
task write_bias(
    input [3:0] address,
    input signed [31:0] data
);
begin
    bias_write_enable = 1;
    bias_write_address = address;
    bias_write_data = data;

    #10;

    bias_write_enable = 0;
end
endtask


integer i;


initial begin

    // ------------------------------------------------------------
    // Initial state
    // ------------------------------------------------------------

    reset = 1;
    start = 0;

    input_write_enable = 0;
    input_write_address = 0;
    input_write_data = 0;

    weight_write_enable = 0;
    weight_write_address = 0;
    weight_write_data = 0;

    bias_write_enable = 0;
    bias_write_address = 0;
    bias_write_data = 0;

    output_read_address = 0;

    #10;

    reset = 0;


// ------------------------------------------------------------
// Load 8 inputs
// Same values used by python/reference.py
// ------------------------------------------------------------

write_input(0,  2);
write_input(1, -1);
write_input(2,  3);
write_input(3,  4);
write_input(4,  0);
write_input(5,  5);
write_input(6, -2);
write_input(7,  1);


// ------------------------------------------------------------
// Load 128 weights
// Stored neuron-by-neuron:
// neuron 0 -> addresses 0-7
// neuron 1 -> addresses 8-15
// ...
// neuron 15 -> addresses 120-127
// ------------------------------------------------------------

// Neuron 0
write_weight(0,  1);
write_weight(1,  2);
write_weight(2,  0);
write_weight(3, -1);
write_weight(4,  3);
write_weight(5,  1);
write_weight(6,  2);
write_weight(7,  0);

// Neuron 1
write_weight(8,   0);
write_weight(9,   1);
write_weight(10,  2);
write_weight(11,  3);
write_weight(12, -1);
write_weight(13,  0);
write_weight(14,  1);
write_weight(15,  2);

// Neuron 2
write_weight(16,  2);
write_weight(17, -1);
write_weight(18,  1);
write_weight(19,  0);
write_weight(20,  2);
write_weight(21, -2);
write_weight(22,  1);
write_weight(23,  3);

// Neuron 3
write_weight(24,  1);
write_weight(25,  0);
write_weight(26, -2);
write_weight(27,  2);
write_weight(28,  1);
write_weight(29,  3);
write_weight(30,  0);
write_weight(31, -1);

// Neuron 4
write_weight(32,  3);
write_weight(33,  1);
write_weight(34,  0);
write_weight(35, -1);
write_weight(36,  2);
write_weight(37,  1);
write_weight(38, -2);
write_weight(39,  1);

// Neuron 5
write_weight(40,  1);
write_weight(41, -2);
write_weight(42,  3);
write_weight(43,  1);
write_weight(44,  0);
write_weight(45,  2);
write_weight(46,  1);
write_weight(47, -1);

// Neuron 6
write_weight(48,  0);
write_weight(49,  2);
write_weight(50,  1);
write_weight(51, -2);
write_weight(52,  3);
write_weight(53,  0);
write_weight(54,  1);
write_weight(55,  2);

// Neuron 7
write_weight(56,  2);
write_weight(57,  1);
write_weight(58, -1);
write_weight(59,  3);
write_weight(60,  0);
write_weight(61, -2);
write_weight(62,  2);
write_weight(63,  1);

// Neuron 8
write_weight(64,  1);
write_weight(65,  3);
write_weight(66,  2);
write_weight(67,  0);
write_weight(68, -1);
write_weight(69,  1);
write_weight(70, -2);
write_weight(71,  2);

// Neuron 9
write_weight(72,  2);
write_weight(73,  0);
write_weight(74,  1);
write_weight(75, -1);
write_weight(76,  3);
write_weight(77,  2);
write_weight(78,  1);
write_weight(79, -2);

// Neuron 10
write_weight(80, -1);
write_weight(81,  2);
write_weight(82,  3);
write_weight(83,  1);
write_weight(84,  0);
write_weight(85, -2);
write_weight(86,  2);
write_weight(87,  1);

// Neuron 11
write_weight(88,  3);
write_weight(89, -1);
write_weight(90,  0);
write_weight(91,  2);
write_weight(92,  1);
write_weight(93,  1);
write_weight(94, -2);
write_weight(95,  2);

// Neuron 12
write_weight(96,   1);
write_weight(97,   2);
write_weight(98,  -1);
write_weight(99,   3);
write_weight(100,  2);
write_weight(101,  0);
write_weight(102,  1);
write_weight(103, -2);

// Neuron 13
write_weight(104,  2);
write_weight(105, -2);
write_weight(106,  1);
write_weight(107,  0);
write_weight(108,  3);
write_weight(109,  1);
write_weight(110,  2);
write_weight(111, -1);

// Neuron 14
write_weight(112,  0);
write_weight(113,  1);
write_weight(114,  3);
write_weight(115, -2);
write_weight(116,  1);
write_weight(117,  2);
write_weight(118, -1);
write_weight(119,  3);

// Neuron 15
write_weight(120,  3);
write_weight(121,  0);
write_weight(122, -1);
write_weight(123,  2);
write_weight(124,  1);
write_weight(125, -2);
write_weight(126,  3);
write_weight(127,  1);


// ------------------------------------------------------------
// Load 16 biases
// ------------------------------------------------------------

write_bias(0,   3);
write_bias(1,  -2);
write_bias(2,   1);
write_bias(3,   4);
write_bias(4,  -1);
write_bias(5,   2);
write_bias(6,   0);
write_bias(7,   3);
write_bias(8,   1);
write_bias(9,  -3);
write_bias(10,  2);
write_bias(11,  0);
write_bias(12,  4);
write_bias(13,  1);
write_bias(14, -2);
write_bias(15,  3);

    // ------------------------------------------------------------
    // Start inference
    // ------------------------------------------------------------

    start = 1;

    // Wait until accelerator finishes
    wait(done == 1);

    $display("");
    $display("Inference complete.");
    $display("");


    // ------------------------------------------------------------
    // Read all 16 outputs
    // ------------------------------------------------------------

    for (i = 0; i < 16; i = i + 1) begin

        output_read_address = i;

        #1;

        $display(
            "output[%0d] = %0d",
            i,
            output_read_data
        );
    end


    // Release START so controller can return to IDLE
    start = 0;

    #20;

    $finish;

end

endmodule