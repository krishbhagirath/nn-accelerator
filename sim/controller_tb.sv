module controller_tb;

logic clk = 0;
logic reset;
logic start;

logic busy;
logic done;

logic mac_clear;
logic mac_enable;
logic output_write_enable;

logic [2:0] input_index;
logic [1:0] batch_index;

controller dut (
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

// Generate clock
always #5 clk = ~clk;

initial begin

    reset = 1;
    start = 0;

    #10;

    reset = 0;

    // Start one full inference
    start = 1;

    // Let the controller run long enough to process all 4 batches
    #500;

    // Release start so DONE can return to IDLE
    start = 0;

    #20;

    $finish;
end


// Print controller status every rising clock edge
always @(posedge clk) begin
    $display(
        "time=%0t busy=%0d done=%0d clear=%0d enable=%0d input=%0d batch=%0d",
        $time,
        busy,
        done,
        mac_clear,
        mac_enable,
        input_index,
        batch_index
    );
end

endmodule