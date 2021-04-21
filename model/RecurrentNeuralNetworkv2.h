#ifndef RECURRENT_NEURAL_NETWORKv2_H
#define RECURRENT_NEURAL_NETWORKv2_H

#include <array>
#include <cmath>
#include <vector>
#include <iostream>
#include <cstdlib>
#include <ctime>
#include <random>
#include <assert.h>
#include "RNNBase.h"

#ifndef NUM_INPUTS
#define NUM_INPUTS 12 
#endif

#ifndef NUM_HIDDEN_UNITS
#define NUM_HIDDEN_UNITS 6
#endif

#ifndef NUM_RECURRENT_UNITS
#define NUM_RECURRENT_UNITS 6
#endif

#ifndef NUM_OUTPUTS
#define NUM_OUTPUTS 9
#endif

typedef std::array<std::array<float, NUM_INPUTS + NUM_RECURRENT_UNITS>, NUM_RECURRENT_UNITS> recurrent_weights_t;
typedef std::array<std::array<float, NUM_RECURRENT_UNITS>, NUM_OUTPUTS> output_weights_t;

class RecurrentNeuralNetworkv2: public RNNBase {
    // Model hyperparameters
    float learning_rate;
    float momentum;
    float weight_decay;
    float grad_clip_threshold; 
    float grad_norm_threshold;
    bool enable_gradient_clipping;
    bool enable_gradient_norm_threshold;

    int NUM_TIME_STEPS;

    // Model parameters
    recurrent_weights_t recurrent_layer_weights;
    output_weights_t output_layer_weights;
    std::array<float, NUM_RECURRENT_UNITS> recurrent_layer_biases;
    std::array<float, NUM_OUTPUTS> output_layer_biases;

    // Cached inputs, cell states, and outputs
    std::vector<std::array<float, NUM_INPUTS>> cache_inputs; 
    std::vector<std::array<float, NUM_RECURRENT_UNITS>> cache_recurrent_units;
    std::vector<std::array<float, NUM_OUTPUTS>> cache_outputs;

    /* Cache for immediate output of recurrent layer weight matrix for ReLU 
     * backpropagation */
//    std::vector<std::array<float, NUM_RECURRENT_UNITS>> cache_intermediate_recurrent_outputs;

    // Cached cumulative sum of update values
    recurrent_weights_t cache_recurrent_weight_updates;
    output_weights_t cache_output_weight_updates;
    std::array<float, NUM_RECURRENT_UNITS> cache_recurrent_bias_updates;
    std::array<float, NUM_OUTPUTS> cache_output_bias_updates;

    // Cached previous updates (at end of recurrent loop) for momentum updates 
    recurrent_weights_t prev_recurrent_weight_updates;
    output_weights_t prev_output_weight_updates;
    std::array<float, NUM_RECURRENT_UNITS> prev_recurrent_bias_updates;
    std::array<float, NUM_OUTPUTS> prev_output_bias_updates;

    std::vector<std::array<float, NUM_RECURRENT_UNITS>> cache_recurring_error_term;

    public:
        RecurrentNeuralNetworkv2(float learning_rate, float momentum, float weight_decay,
                float grad_clip_threshold, float grad_norm_threshold,
                bool enable_gradient_clipping,
                bool enable_gradient_norm_threshold);
        float train(std::vector<std::array<float, NUM_INPUTS>> &sequence,
                int truth_label);
        int inference(std::vector<std::array<float, NUM_INPUTS>> &sequence);
        void save_weights(std::string filename);
        void load_weights(std::string filename);
        ~RecurrentNeuralNetworkv2() {}

    private:
        // Activation functions
        float sigmoid(float x);
        float leaky_relu(float x, float alpha);
        float leaky_relu_backprop(float i);
        float softmax(float x, std::array<float, NUM_OUTPUTS> &x_vector);

        float cross_entropy_loss(std::array<float, NUM_OUTPUTS> &preds,
                        std::array<float, NUM_OUTPUTS> &targets);

        void initialize_weights();
        std::array<float, NUM_OUTPUTS> cell_forward(
                            std::array<float, NUM_INPUTS> &inputs,
                            bool train = false);
        std::array<float, NUM_RECURRENT_UNITS> cell_backpropagation(
                std::array<float, NUM_OUTPUTS> &targets,
                std::array<float, NUM_OUTPUTS> &preds,
                std::array<float, NUM_RECURRENT_UNITS> &next_grad,
                bool last_time_step = false);

        void perform_weight_updates();
        void clear_cache();

        std::array<float, NUM_OUTPUTS> one_hot_encoding(int truth_label);

        std::array<float, NUM_RECURRENT_UNITS> gradient_clipping(
                std::array<float, NUM_RECURRENT_UNITS> &gradient);

        std::array<float, NUM_RECURRENT_UNITS> gradient_norm_scaling(
                std::array<float, NUM_RECURRENT_UNITS> &gradient);
};


#endif 
