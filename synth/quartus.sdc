# 50 MHz, the DE2-115 board oscillator. Quartus still reports the real Fmax.
create_clock -name clk -period 20.000 [get_ports clk]
