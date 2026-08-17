module output_memory_tb;

logic clk = 0;
logic write_enable;

logic [3:0] write_address0;
logic [3:0] write_address1;
logic [3:0] write_address2;
logic [3:0] write_address3;

logic signed [7:0] write_data0;
logic signed [7:0] write_data1;
logic signed [7:0] write_data2;
logic signed [7:0] write_data3;

logic [3:0] read_address;
logic signed [7:0] read_data;


// Instantiate output memory
output_memory dut (
    .clk(clk),
    .write_enable(write_enable),

    .write_address0(write_address0),
    .write_address1(write_address1),
    .write_address2(write_address2),
    .write_address3(write_address3),

    .write_data0(write_data0),
    .write_data1(write_data1),
    .write_data2(write_data2),
    .write_data3(write_data3),

    .read_address(read_address),
    .read_data(read_data)
);


// Generate clock
always #5 clk = ~clk;


initial begin

    write_enable = 0;

    write_address0 = 0;
    write_address1 = 0;
    write_address2 = 0;
    write_address3 = 0;

    write_data0 = 0;
    write_data1 = 0;
    write_data2 = 0;
    write_data3 = 0;

    read_address = 0;

    #10;


    // Write four neuron outputs simultaneously
    write_enable = 1;

    write_address0 = 0;
    write_address1 = 1;
    write_address2 = 2;
    write_address3 = 3;

    write_data0 = 25;
    write_data1 = 50;
    write_data2 = 75;
    write_data3 = 100;

    #10;

    write_enable = 0;


    // Read the stored outputs one at a time
    read_address = 0;
    #1;
    $display("output[0] = %0d (expected 25)", read_data);

    read_address = 1;
    #1;
    $display("output[1] = %0d (expected 50)", read_data);

    read_address = 2;
    #1;
    $display("output[2] = %0d (expected 75)", read_data);

    read_address = 3;
    #1;
    $display("output[3] = %0d (expected 100)", read_data);

    $finish;

end

endmodule