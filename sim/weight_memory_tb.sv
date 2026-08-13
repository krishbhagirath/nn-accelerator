module weight_memory_tb;

logic clk = 0;

logic write_enable;
logic [6:0] write_address;
logic signed [7:0] write_data;

logic [6:0] read_address0;
logic [6:0] read_address1;
logic [6:0] read_address2;
logic [6:0] read_address3;

logic signed [7:0] read_data0;
logic signed [7:0] read_data1;
logic signed [7:0] read_data2;
logic signed [7:0] read_data3;

weight_memory dut (
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

    // Load four weights
    write_enable = 1;

    write_address = 0;
    write_data = 3;
    #10;

    write_address = 8;
    write_data = -2;
    #10;

    write_address = 16;
    write_data = 5;
    #10;

    write_address = 24;
    write_data = 7;
    #10;

    write_enable = 0;

    // Read all four at the same time
    read_address0 = 0;
    read_address1 = 8;
    read_address2 = 16;
    read_address3 = 24;

    #1;

    $display("read_data0 = %0d (expected 3)", read_data0);
    $display("read_data1 = %0d (expected -2)", read_data1);
    $display("read_data2 = %0d (expected 5)", read_data2);
    $display("read_data3 = %0d (expected 7)", read_data3);

    $finish;

end

endmodule