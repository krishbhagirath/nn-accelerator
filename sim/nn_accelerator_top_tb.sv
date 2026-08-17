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
    //
    // [2, 5, 1, 7, 3, 4, 6, 8]
    // ------------------------------------------------------------

    write_input(0, 2);
    write_input(1, 5);
    write_input(2, 1);
    write_input(3, 7);
    write_input(4, 3);
    write_input(5, 4);
    write_input(6, 6);
    write_input(7, 8);


    // ------------------------------------------------------------
    // Load 128 weights
    //
    // For this first integration test, every neuron uses:
    // [1, 1, 1, 1, 1, 1, 1, 1]
    //
    // So every neuron should compute the same weighted sum.
    // ------------------------------------------------------------

    for (i = 0; i < 128; i = i + 1) begin
        write_weight(i, 1);
    end


    // ------------------------------------------------------------
    // Load 16 biases
    //
    // Use bias = neuron index
    // neuron 0 bias = 0
    // neuron 1 bias = 1
    // ...
    // neuron 15 bias = 15
    // ------------------------------------------------------------

    for (i = 0; i < 16; i = i + 1) begin
        write_bias(i, i);
    end


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