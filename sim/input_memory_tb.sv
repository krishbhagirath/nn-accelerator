module input_memory_tb;

logic clk = 0;

logic write_enable;
logic [2:0] write_address;
logic signed [7:0] write_data;

logic [2:0] read_address;
logic signed [7:0] read_data;

input_memory dut (
    .clk(clk),
    .write_enable(write_enable),
    .write_address(write_address),
    .write_data(write_data),
    .read_address(read_address),
    .read_data(read_data)
);

always #5 clk = ~clk;

initial begin

    write_enable = 0;
    write_address = 0;
    write_data = 0;
    read_address = 0;

    #10;

    // Write memory[0] = 2
    write_enable = 1;
    write_address = 0;
    write_data = 2;
    #10;

    // Write memory[1] = 5
    write_address = 1;
    write_data = 5;
    #10;

    // Write memory[2] = 1
    write_address = 2;
    write_data = 1;
    #10;

    // Write memory[3] = 7
    write_address = 3;
    write_data = 7;
    #10;

    write_enable = 0;

    // Read them back
    read_address = 0;
    #1;
    $display("memory[0] = %0d (expected 2)", read_data);

    read_address = 1;
    #1;
    $display("memory[1] = %0d (expected 5)", read_data);

    read_address = 2;
    #1;
    $display("memory[2] = %0d (expected 1)", read_data);

    read_address = 3;
    #1;
    $display("memory[3] = %0d (expected 7)", read_data);

    $finish;

end

endmodule