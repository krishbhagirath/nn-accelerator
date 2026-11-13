// C++ runtime for the Verilated accelerator.
//
// Does exactly what a driver for the real chip would do, using the same steps
// as the cocotb driver (sim/accelerator.py):
//
//   for each test sample:
//       for each layer:
//           load weights, biases and inputs   (one value per clock)
//           set input_dim / output_dim / shift / relu
//           raise start, clock until done, lower start
//           read the outputs                  -> inputs of the next layer
//       prediction = argmax(final outputs)
//
// Usage:  Vnn_accelerator_top <compiled model dir> [max samples]
// NUM_MACS is passed at compile time with -DNUM_MACS=<n> (see run_pipeline.py).

#include <cstdint>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "Vnn_accelerator_top.h"
#include "verilated.h"

#ifndef NUM_MACS
#define NUM_MACS 4
#endif

using Matrix = std::vector<std::vector<int64_t>>;

// Verilator's library still links against this legacy simulation-time hook.
// The design has no delays, so simulation time is never used.
double sc_time_stamp() { return 0; }

struct Layer {
    int input_dim, output_dim, output_shift;
    bool relu;
    Matrix weights;               // [output_dim][input_dim]
    std::vector<int64_t> biases;  // [output_dim]
};

// ------------------------------------------------------------
// File reading
// ------------------------------------------------------------

// Every compiler output file is whitespace-separated integers, one row per line.
static Matrix read_matrix(const std::string& path) {
    std::ifstream file(path);
    if (!file) { std::cerr << "cannot open " << path << "\n"; std::exit(1); }
    Matrix rows;
    std::string line;
    while (std::getline(file, line)) {
        std::istringstream values(line);
        std::vector<int64_t> row;
        int64_t v;
        while (values >> v) row.push_back(v);
        if (!row.empty()) rows.push_back(row);
    }
    return rows;
}

// Minimal reader for OUR manifest.json: find the next "key": after `pos` and
// return its value. We wrote the file ourselves, so we know its exact shape;
// this avoids depending on a JSON library.
static std::string json_value(const std::string& text, const std::string& key, size_t& pos) {
    pos = text.find("\"" + key + "\":", pos);
    if (pos == std::string::npos) { std::cerr << "manifest missing " << key << "\n"; std::exit(1); }
    pos += key.size() + 3;
    size_t end = text.find_first_of(",}\n", pos);
    std::string value = text.substr(pos, end - pos);
    value.erase(0, value.find_first_not_of(" \""));
    value.erase(value.find_last_not_of(" \"") + 1);
    return value;
}

static std::vector<Layer> load_model(const std::string& dir, int& num_samples) {
    std::ifstream file(dir + "/manifest.json");
    std::stringstream buffer;
    buffer << file.rdbuf();
    const std::string text = buffer.str();

    size_t pos = 0;
    num_samples = std::stoi(json_value(text, "num_test_samples", pos));
    const int num_layers = std::stoi(json_value(text, "num_layers", pos = 0));

    std::vector<Layer> layers(num_layers);
    pos = text.find("\"layers\"");
    for (Layer& layer : layers) {
        layer.input_dim = std::stoi(json_value(text, "input_dim", pos));
        layer.output_dim = std::stoi(json_value(text, "output_dim", pos));
        layer.relu = json_value(text, "relu", pos) == "true";
        layer.output_shift = std::stoi(json_value(text, "output_shift", pos));
        layer.weights = read_matrix(dir + "/" + json_value(text, "weights_file", pos));
        for (auto& row : read_matrix(dir + "/" + json_value(text, "biases_file", pos)))
            layer.biases.push_back(row[0]);
    }
    return layers;
}

// ------------------------------------------------------------
// Driving the hardware
// ------------------------------------------------------------

struct Accelerator {
    VerilatedContext context;
    Vnn_accelerator_top top{&context};
    uint64_t cycles = 0;  // every clock cycle the host spends

    // One full clock cycle: inputs set before this are sampled at the rising edge.
    void tick() {
        top.clk = 0; top.eval();
        top.clk = 1; top.eval();
        cycles++;
    }

    void reset() {
        top.reset = 1; top.start = 0;
        top.input_write_enable = top.weight_write_enable = top.bias_write_enable = 0;
        tick(); tick();
        top.reset = 0;
        tick();
    }

    void load_layer(const Layer& layer) {
        for (int n = 0; n < layer.output_dim; n++) {
            for (int i = 0; i < layer.input_dim; i++) {
                top.weight_write_enable = 1;
                top.weight_write_neuron = n;
                top.weight_write_input = i;
                top.weight_write_data = static_cast<uint8_t>(layer.weights[n][i]);
                tick();
            }
            top.weight_write_enable = 0;
            top.bias_write_enable = 1;
            top.bias_write_neuron = n;
            top.bias_write_data = static_cast<uint32_t>(layer.biases[n]);
            tick();
            top.bias_write_enable = 0;
        }
    }

    void load_inputs(const std::vector<int64_t>& inputs) {
        for (size_t i = 0; i < inputs.size(); i++) {
            top.input_write_enable = 1;
            top.input_write_address = i;
            top.input_write_data = static_cast<uint8_t>(inputs[i]);
            tick();
        }
        top.input_write_enable = 0;
    }

    // Returns the hardware cycle_count for this layer
    uint32_t run(const Layer& layer) {
        top.config_input_dim = layer.input_dim;
        top.config_output_dim = layer.output_dim;
        top.config_output_shift = layer.output_shift;
        top.config_relu = layer.relu;
        top.start = 1;
        do { tick(); } while (!top.done);
        const uint32_t compute_cycles = top.cycle_count;
        top.start = 0;
        tick();
        return compute_cycles;
    }

    std::vector<int64_t> read_outputs(int count) {
        std::vector<int64_t> outputs;
        for (int n = 0; n < count; n++) {
            top.output_read_address = n;
            top.eval();  // combinational read
            outputs.push_back(static_cast<int8_t>(top.output_read_data));
            tick();
        }
        return outputs;
    }
};

static int argmax(const std::vector<int64_t>& values) {
    int best = 0;
    for (size_t i = 1; i < values.size(); i++)
        if (values[i] > values[best]) best = static_cast<int>(i);
    return best;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: " << argv[0] << " <compiled model dir> [max samples]\n";
        return 1;
    }
    const std::string dir = argv[1];
    int num_samples = 0;
    std::vector<Layer> layers = load_model(dir, num_samples);
    if (argc > 2) num_samples = std::min(num_samples, std::atoi(argv[2]));

    const Matrix inputs = read_matrix(dir + "/test_inputs.txt");
    const Matrix expected = read_matrix(dir + "/expected_outputs.txt");
    const Matrix labels = read_matrix(dir + "/test_labels.txt");

    Accelerator accel;
    accel.reset();

    int bit_exact = 0, correct = 0;
    uint64_t compute_cycles = 0, mac_operations = 0;
    for (int s = 0; s < num_samples; s++) {
        std::vector<int64_t> activations = inputs[s];
        compute_cycles = 0;
        mac_operations = 0;
        for (const Layer& layer : layers) {
            accel.load_layer(layer);
            accel.load_inputs(activations);
            compute_cycles += accel.run(layer);
            mac_operations += accel.top.mac_operations;
            activations = accel.read_outputs(layer.output_dim);
        }
        const bool exact = activations == expected[s];
        const int prediction = argmax(activations);
        bit_exact += exact;
        correct += prediction == labels[s][0];
        if (s < 5)
            std::cout << "sample " << s << ": label=" << labels[s][0] << " RTL=" << prediction
                      << " golden=" << argmax(expected[s]) << "  " << (exact ? "PASS" : "FAIL") << "\n";
    }

    std::cout << "\nmodel dir:            " << dir << "\n"
              << "NUM_MACS:             " << NUM_MACS << "\n"
              << "samples:              " << num_samples << "\n"
              << "bit-exact vs golden:  " << bit_exact << "/" << num_samples << "\n"
              << "RTL accuracy:         " << 100.0 * correct / num_samples << "%\n"
              << "compute cycles/inference (hardware counter): " << compute_cycles << "\n"
              << "MAC utilization:      "
              << 100.0 * mac_operations / (NUM_MACS * compute_cycles) << "%\n"
              << "total simulated host cycles (incl. weight reloads): " << accel.cycles << "\n"
              << (bit_exact == num_samples ? "RESULT: PASS" : "RESULT: FAIL") << "\n";
    return bit_exact == num_samples ? 0 : 1;
}
