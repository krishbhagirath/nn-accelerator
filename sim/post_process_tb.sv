module post_process_tb;

logic signed [31:0] acc0, acc1, acc2, acc3;
logic signed [31:0] bias0, bias1, bias2, bias3;

logic signed [7:0] out0, out1, out2, out3;

post_process dut (
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

initial begin

    // Test normal value
    // 40 + 10 = 50
    acc0 = 40;
    bias0 = 10;

    // Test negative result -> ReLU should make it 0
    // -20 + 5 = -15 -> 0
    acc1 = -20;
    bias1 = 5;

    // Test saturation
    // 120 + 20 = 140 -> 127
    acc2 = 120;
    bias2 = 20;

    // Test another normal value
    // 30 + (-10) = 20
    acc3 = 30;
    bias3 = -10;

    #1;

    $display("out0 = %0d (expected 50)", out0);
    $display("out1 = %0d (expected 0)", out1);
    $display("out2 = %0d (expected 127)", out2);
    $display("out3 = %0d (expected 20)", out3);

    $finish;

end

endmodule