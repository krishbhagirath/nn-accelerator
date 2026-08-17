// Controls the accelerator's execution flow.
// Tracks which input and neuron batch are being processed.
// Generates the high-level control signals for the MAC array.

module controller (
    input  logic clk,
    input  logic reset,
    input  logic start,

    output logic busy,
    output logic done,

    output logic mac_clear,
    output logic mac_enable,
    output logic output_write_enable,

    output logic [2:0] input_index,  // 0-7 inputs
    output logic [1:0] batch_index  // 0-3 batches of 4 neurons
);

    // FSM states
    typedef enum logic [2:0] {
        IDLE,
        CLEAR,
        COMPUTE,
        POST_PROCESS,
        WRITE_OUTPUTS,
        NEXT_BATCH,
        DONE
    } state_t;

    state_t current_state;
    state_t next_state;


    // ------------------------------------------------------------
    // State register
    // Moves the FSM to next_state on each rising clock edge.
    // Reset always returns the controller to IDLE.
    // ------------------------------------------------------------
    always_ff @(posedge clk) begin
        if (reset)
            current_state <= IDLE;
        else
            current_state <= next_state;
    end


    // ------------------------------------------------------------
    // Counters
    // input_index tracks which of the 8 inputs is being processed.
    // batch_index tracks which group of 4 neurons is being processed.
    // ------------------------------------------------------------
    always_ff @(posedge clk) begin
        if (reset) begin
            input_index <= 0;
            batch_index <= 0;
        end

        else begin

            // Start each new batch from input 0
            if (current_state == CLEAR)
                input_index <= 0;

            // During COMPUTE, move through inputs 0 -> 7
            else if (current_state == COMPUTE) begin
                if (input_index < 7)
                    input_index <= input_index + 1;
            end

            // After finishing a batch, move to the next group of 4 neurons
            if (current_state == NEXT_BATCH) begin
                if (batch_index < 3)
                    batch_index <= batch_index + 1;
            end

            // New inference starts from batch 0
            if (current_state == IDLE && start)
                batch_index <= 0;

        end
    end


    // ------------------------------------------------------------
    // Next-state logic
    // Decides which state the controller should enter next.
    // ------------------------------------------------------------
    always_comb begin

        // Default: stay in the current state
        next_state = current_state;

        case (current_state)

            IDLE: begin
                if (start)
                    next_state = CLEAR;
            end


            // Clear MAC accumulators before starting a neuron batch
            CLEAR: begin
                next_state = COMPUTE;
            end


            // Process all 8 inputs for the current 4-neuron batch
            COMPUTE: begin
                if (input_index == 7)
                    next_state = POST_PROCESS;
            end


            // Bias + ReLU + saturation happen here
            POST_PROCESS: begin
                next_state = WRITE_OUTPUTS;
            end


            // Store the four completed neuron outputs
            WRITE_OUTPUTS: begin
                next_state = NEXT_BATCH;
            end


            // If this was batch 3, all 16 neurons are complete
            NEXT_BATCH: begin
                if (batch_index == 3)
                    next_state = DONE;
                else
                    next_state = CLEAR;
            end


            // Hold DONE until start is released
            DONE: begin
                if (!start)
                    next_state = IDLE;
            end


            default: begin
                next_state = IDLE;
            end

        endcase
    end


    // ------------------------------------------------------------
    // Output/control logic
    // These signals depend only on the current FSM state.
    // ------------------------------------------------------------
    always_comb begin

        // Safe/default values
        busy                = 0;
        done                = 0;
        mac_clear           = 0;
        mac_enable          = 0;
        output_write_enable = 0;

        case (current_state)

            IDLE: begin
                // Waiting for START
            end

            CLEAR: begin
                busy      = 1;
                mac_clear = 1;
            end

            COMPUTE: begin
                busy       = 1;
                mac_enable = 1;
            end

            POST_PROCESS: begin
                busy = 1;
            end

            WRITE_OUTPUTS: begin
                busy = 1;
                output_write_enable = 1;
            end

            NEXT_BATCH: begin
                busy = 1;
            end

            DONE: begin
                done = 1;
            end

            default: begin
                // Keep defaults
            end

        endcase
    end

endmodule