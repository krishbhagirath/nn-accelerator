module bias_memory_tb;

logic clk = 0;

logic write_enable;
logic [3:0] write_address;
logic signed [31:0] write_data;

logic [3:0] read_address0;
logic [3:0] read_address1;
logic [3:0] read_address2;
logic [3:0] read_address3;

logic signed [31:0] read_data0;
logic signed [31:0] read_data1;
logic signed [31:0] read_data2;
logic signed [31:0] read_data3;

bias_memory dut (
    .clk(clk),

    .write_enable(write_enable),
    .write_address(write_address),
    .write_data(write_data),

    .read_address0(read_address0),
    .read_address1(read_address1),
    .read_address2(read_address2),
    .read_address3(read_address3),

    .read_data0(read_data0),
    .read_data1(read_data1),
    .read_data2(read_data2),
    .read_data3(read_data3)
);

always #5 clk = ~clk;

initial begin
    write_enable = 0;
    write_address = 0;
    write_data = 0;

    read_address0 = 0;
    read_address1 = 0;
    read_address2 = 0;
    read_address3 = 0;

    #10;

    // Load four biases
    write_enable = 1;

    write_address = 0;
    write_data = 10;
    #10;

    write_address = 1;
    write_data = -4;
    #10;

    write_address = 2;
    write_data = 7;
    #10;

    write_address = 3;
    write_data = 2;
    #10;

    write_enable = 0;

    // Read all four in parallel
    read_address0 = 0;
    read_address1 = 1;
    read_address2 = 2;
    read_address3 = 3;

    #1;

    $display("bias0 = %0d (expected 10)", read_data0);
    $display("bias1 = %0d (expected -4)", read_data1);
    $display("bias2 = %0d (expected 7)", read_data2);
    $display("bias3 = %0d (expected 2)", read_data3);

    $finish;
end

endmodule