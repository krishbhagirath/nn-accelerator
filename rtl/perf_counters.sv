// Hardware performance counters for one layer run.
//
// All counters reset to 0 when a run starts and then hold their final values
// after done, so the host can read them at any time.
//
//   cycle_count       cycles spent busy (COMPUTE + DRAIN states). This is the
//                     layer's compute latency, NOT including the 1 cycle to
//                     accept start or any host load/read cycles.
//   mac_active_cycles cycles in which the MAC lanes were enabled.
//   mac_operations    useful multiply-accumulates: active cycles x lanes that
//                     hold a real neuron (unused lanes of a partial batch do
//                     not count). Always equals input_dim * output_dim.
//
// Utilization = mac_operations / (NUM_MACS * cycle_count), computed in software.

`timescale 1ns/1ps

module perf_counters #(
    parameter NUM_MACS = 4
)(
    input  logic                clk,
    input  logic                reset,
    input  logic                run_started,
    input  logic                busy,
    input  logic                mac_enable,
    input  logic [NUM_MACS-1:0] mac_lane_valid,

    output logic [31:0]         cycle_count,
    output logic [31:0]         mac_active_cycles,
    output logic [31:0]         mac_operations
);

    // How many lanes are doing useful work this cycle
    logic [31:0] useful_lanes;
    always_comb begin
        useful_lanes = 0;
        for (int lane = 0; lane < NUM_MACS; lane++)
            useful_lanes = useful_lanes + 32'(mac_lane_valid[lane]);
    end

    always_ff @(posedge clk) begin
        if (reset || run_started) begin
            cycle_count       <= 0;
            mac_active_cycles <= 0;
            mac_operations    <= 0;
        end else begin
            if (busy)
                cycle_count <= cycle_count + 1;
            if (mac_enable) begin
                mac_active_cycles <= mac_active_cycles + 1;
                mac_operations    <= mac_operations + useful_lanes;
            end
        end
    end

endmodule
