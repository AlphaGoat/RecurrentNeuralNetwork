#include <string>
#include <fstream>
#include <sstream>
#include "RNNBase.h"
#include "RecurrentNeuralNetworkv2.h"

#pragma GCC diagnostic ignored "-Wsign-compare"

#define OFFSET 1e-4

#define ALPHA 1e-2 // For leaky ReLU

std::default_random_engine rnnv2_generator;

RecurrentNeuralNetworkv2::RecurrentNeuralNetworkv2(
        float learning_rate, float momentum, float weight_decay,
        float grad_clip_threshold, float grad_norm_threshold,
        bool enable_gradient_clipping, bool enable_gradient_norm_threshold,
        bool He_initialization, bool normal_weight_updates) {
    RecurrentNeuralNetworkv2::learning_rate = learning_rate;
    RecurrentNeuralNetworkv2::momentum = momentum;
    RecurrentNeuralNetworkv2::weight_decay = weight_decay;
    RecurrentNeuralNetworkv2::grad_clip_threshold = grad_clip_threshold;
    RecurrentNeuralNetworkv2::grad_norm_threshold = grad_norm_threshold;
    RecurrentNeuralNetworkv2::enable_gradient_clipping = enable_gradient_clipping;
    RecurrentNeuralNetworkv2::enable_gradient_norm_threshold = enable_gradient_norm_threshold;
    RecurrentNeuralNetworkv2::He_initialization = He_initialization;
    RecurrentNeuralNetworkv2::NORMAL_UPDATES_FLAG = normal_weight_updates;
    RecurrentNeuralNetworkv2::initialize_weights();
}

void RecurrentNeuralNetworkv2::initialize_weights() {
    /* Initialize weights of network to uniform distribution 
     * in range [-1.0, 1.0] */

    srand( (unsigned int) time(NULL));
    std::uniform_real_distribution<float> uniform_distribution(-1, 1);


    // If using He initialization scheme, the distribution for each layer 
    // is different (and dependent on the number of inputs to the layer)
    
    // He initialization STDDEV: sqrt(2 / NUM_RECURRENT_UNITS + NUM_INPUTS)
    std::normal_distribution<float> He_distribution_recurrent(0.0, 
            std::sqrt(2.0 / (NUM_INPUTS + NUM_RECURRENT_UNITS)));

    // Initialize Recurrent weights 
    for (int i = 0; i < NUM_RECURRENT_UNITS; i++) {
        for (int j = 0; j < NUM_INPUTS + NUM_RECURRENT_UNITS; j++) {

            if (He_initialization) {
                recurrent_layer_weights[i][j] = He_distribution_recurrent(rnnv2_generator);
            }
            else {
                recurrent_layer_weights[i][j] = uniform_distribution(rnnv2_generator);
            }
        }

        if (He_initialization) {
            recurrent_layer_biases[i] = He_distribution_recurrent(rnnv2_generator);
        }
        else {
            recurrent_layer_biases[i] = uniform_distribution(rnnv2_generator);
        }
    }

    // He initialization STDDEV: sqrt(2 / NUM_RECURRENT_UNITS)
    std::normal_distribution<float> He_distribution_output(0.0, 
            std::sqrt(2.0 / (NUM_RECURRENT_UNITS)));

    // Initialize output weights
    for (int i = 0; i < NUM_OUTPUTS; i++) {
        for (int j = 0; j < NUM_RECURRENT_UNITS; j++) {

            if (He_initialization) {
                output_layer_weights[i][j] = He_distribution_output(rnnv2_generator);
            }

            else {
                output_layer_weights[i][j] = uniform_distribution(rnnv2_generator);
            }
        }

        if (He_initialization) {
            output_layer_biases[i] = He_distribution_output(rnnv2_generator);
        }

        else {
            output_layer_biases[i] = uniform_distribution(rnnv2_generator);
        }

    }

    // initialize first instance in cache of 
    // recurrent neural network to zero
    std::array<float, NUM_RECURRENT_UNITS> zeros;
    for (int i = 0; i < NUM_RECURRENT_UNITS; i++) {
        zeros[i] = 0;
    }
    cache_recurrent_units.push_back(zeros);

    // initialize weight updates to zero 
    for (int i = 0; i < NUM_RECURRENT_UNITS; i++) {
        for (int j = 0; j < NUM_INPUTS + NUM_RECURRENT_UNITS; j++) {
            cache_recurrent_weight_updates[i][j] = 0.0;
        }
        cache_recurrent_bias_updates[i] = 0.0;
    }

    for (int i = 0; i < NUM_OUTPUTS; i++) {
        for (int j = 0; j < NUM_RECURRENT_UNITS; j++) {
            cache_output_weight_updates[i][j] = 0.0;
        }
        cache_output_bias_updates[i] = 0.0;
    }

    // Initialize "previous" weight update to zero 
    for (int i = 0; i < NUM_RECURRENT_UNITS; i++) {
        for (int j = 0; j < NUM_INPUTS + NUM_RECURRENT_UNITS; j++) {
            prev_recurrent_weight_updates[i][j] = 0.0;
        }

        prev_recurrent_bias_updates[i] = 0.0;
    }

    for (int i = 0; i < NUM_OUTPUTS; i++) {
        for (int j = 0; j < NUM_RECURRENT_UNITS; j++) {
            prev_output_weight_updates[i][j] = 0.0;
        }

        prev_output_bias_updates[i] = 0.0;
    }
}

std::array<float, NUM_OUTPUTS> RecurrentNeuralNetworkv2::cell_forward(
        std::array<float, NUM_INPUTS> &inputs,
        bool train) {

    /* Retrieve recurrent units from previous sequence */
    std::array<float, NUM_RECURRENT_UNITS> prev_state = cache_recurrent_units.back();

    /* concatenated array of recurrent units and inputs */ 
    std::array<float, NUM_RECURRENT_UNITS + NUM_INPUTS> concat_inputs;

    for (int i = 0; i < NUM_INPUTS; i++) 
        concat_inputs[i] = inputs[i];
    
    for (int i = 0; i < NUM_RECURRENT_UNITS; i++)
        concat_inputs[NUM_INPUTS + i] = prev_state[i];

    // Perform forward pass for recurrent layer
    std::array<float, NUM_RECURRENT_UNITS> current_state;
//    std::array<float, NUM_RECURRENT_UNITS> intermediate_output;
    for (int i = 0; i < NUM_RECURRENT_UNITS; i++) {
        float recurrent_i = 0.0;
        for (int j = 0; j < NUM_INPUTS + NUM_RECURRENT_UNITS; j++) {
            recurrent_i += recurrent_layer_weights[i][j] * concat_inputs[j]; 
        }

        recurrent_i += recurrent_layer_biases[i];
//        intermediate_output[i] = recurrent_i;

        // leaky relu activation 
        current_state[i] = RecurrentNeuralNetworkv2::leaky_relu(recurrent_i, ALPHA);

        if (std::isnan(current_state[i])) {
            int x = 5;
            int y = 2;
        }
    }
//    cache_intermediate_recurrent_outputs.push_back(intermediate_output);
    
    /* Calculate outputs */
    std::array<float, NUM_OUTPUTS> outputs;
    for (int i = 0; i < NUM_OUTPUTS; i++) {
        float output_i = 0.0;
        for (int j = 0; j < NUM_RECURRENT_UNITS; j++) {
            output_i += output_layer_weights[i][j] * current_state[j];
        }

        output_i += output_layer_biases[i];
        outputs[i] = output_i;

        if (std::isnan(outputs[i])) {
            int x = 5;
            int y = 2;
        }
    }

    /* Apply softmax activation */
    std::array<float, NUM_OUTPUTS> activated_outputs;
    for (int i = 0; i < NUM_OUTPUTS; i++) {
        activated_outputs[i] = RecurrentNeuralNetworkv2::softmax(outputs[i], outputs);

        if (std::isnan(activated_outputs[i])) {
            int x = 5;
            int y = 2;
        }
//        if (std::isnan(activated_outputs[i])) {
//            std::cout << " activated output is NAN\n";
//            std::cout << " do something\n";
//        }
    }

    // Cache intermediate layer ouputs 
    if (train == true) {
        cache_inputs.push_back(inputs);
        cache_recurrent_units.push_back(current_state);
        cache_outputs.push_back(activated_outputs);
    }

    // return outputs 
    return activated_outputs;
}

std::array<float, NUM_RECURRENT_UNITS> RecurrentNeuralNetworkv2::cell_backpropagation(
                std::array<float, NUM_OUTPUTS> &targets,
                std::array<float, NUM_OUTPUTS> &preds,
                std::array<float, NUM_RECURRENT_UNITS> &next_cell_grad,
                bool last_time_step) {

    // Perform backpropagation and cache weight updates 
    std::array<float, NUM_OUTPUTS> layer_outputs = cache_outputs.back();
    std::array<float, NUM_RECURRENT_UNITS> current_state = cache_recurrent_units.back();
    cache_outputs.pop_back();
    cache_recurrent_units.pop_back();

    std::array<float, NUM_INPUTS> input = cache_inputs.back();
    std::array<float, NUM_RECURRENT_UNITS> prev_state = cache_recurrent_units.back();
    cache_inputs.pop_back();

    /* Calculate output error term for this step */
    /* NOTE: error term is the negative of the loss gradient 
     * with respect to the outputs of the output layer */
    std::array<float, NUM_OUTPUTS> output_error_terms;
    for (int j = 0; j < NUM_OUTPUTS; j++) {
        /* For softmax / cross-entropy loss, the 
         * error gradient wrt layer output is just 
         * (o - t), where o is activavted output and 
         * t is the target. So error term is -(o - t)
         *  = (t - o) */
        output_error_terms[j] = (targets[j] - layer_outputs[j]);
    }

    /* Clip gradient 

    for (int i = 0; i < NUM_OUTPUTS; i++ ) {
        for (int k = 0; k < NUM_RECURRENT_UNITS; k++) {
            cache_output_weight_updates[i][k] += learning_rate * output_error_terms[i]
                        * current_state[k];
        }

        cache_output_bias_updates[i] += learning_rate * output_error_terms[i];
    }

    /* Calculate recurrent error term for this time step */
    /* NOTE: the recurrent error term at this time step is 
     * dependent on all steps ahead of this, as the outputs of
     * this recurrent layer influence all of the outputs of 
     * subsequent layers */
    std::array<float, NUM_RECURRENT_UNITS> current_state_error_grads; // error gradients for 
                                                                      // this cell's state
    /* For ReLU backprop, need to get intermediate output of recurrent layer 
     * at this time step before activation (piecewise function, so derivate calc
     * changes based on input */
    for (int i = 0; i < NUM_RECURRENT_UNITS; i++) {
        float current_error_term = 0.0; /* error term for current step's output */
        float recurring_error_term = 0.0; /* recurring error term from steps after this */

        float last_step_term = 0.0;
        for (int j = 0; j < NUM_OUTPUTS; j++) {
            // dO / dH
            current_error_term += output_error_terms[j] * output_layer_weights[j][i];

        }

        for (int k = 0; k < NUM_RECURRENT_UNITS; k++) {
            // dHt / dHt-1
            recurring_error_term += next_cell_grad[k] * recurrent_layer_weights[k][i];

        }

        current_state_error_grads[i] = leaky_relu_backprop(current_state[i]) 
                            * (current_error_term + recurring_error_term);

    }

    /* Clip gradient if enabled */
    if (enable_gradient_clipping) {
        current_state_error_grads = gradient_clipping(current_state_error_grads);
    }

    if (enable_gradient_norm_threshold) {
        current_state_error_grads = gradient_norm_scaling(current_state_error_grads);
    }


    /* Concatenate inputs */
    std::array<float, NUM_INPUTS + NUM_RECURRENT_UNITS> concat_inputs;
    for (int i = 0; i < NUM_INPUTS; i++) 
        concat_inputs[i] = input[i];

    for (int j = 0; j < NUM_RECURRENT_UNITS; j++) 
        concat_inputs[NUM_INPUTS + j] = prev_state[j];

    /* Perform updates for recurrent layer */
    for (int j = 0; j < NUM_RECURRENT_UNITS; j++) {
        for (int k = 0; k < NUM_INPUTS + NUM_RECURRENT_UNITS; k++) {
            cache_recurrent_weight_updates[j][k] += learning_rate
                    * current_state_error_grads[j] * concat_inputs[k];
        }

        cache_recurrent_bias_updates[j] += learning_rate * current_state_error_grads[j];

    }

    return current_state_error_grads;
        
}

void RecurrentNeuralNetworkv2::perform_weight_updates() {

    // Get the number of time steps in the sequence

    /* Perform updates for output weights and biases */
    for (int i = 0; i < NUM_OUTPUTS; i++) {
        for (int j = 0; j < NUM_RECURRENT_UNITS; j++) {
            /* weight decay */
            float weight_decay_term = -2 * weight_decay * learning_rate * 
                        output_layer_weights[i][j];

            // Mean of the update over all time steps
            float mean_weight_update = cache_output_weight_updates[i][j] / NUM_TIME_STEPS;

            /* weight update */
            output_layer_weights[i][j] += mean_weight_update 
                                + (momentum * prev_output_weight_updates[i][j])
                                + weight_decay_term;

            /* Store updates for momentum calulation in next iteration */
            prev_output_weight_updates[i][j] = mean_weight_update 
                                + (momentum * prev_output_weight_updates[i][j])
                                + weight_decay_term;
        }

        /* Weight decay */
        float weight_decay_term = -2 * weight_decay * learning_rate * 
                    output_layer_biases[i];

        /* mean of the update over all time steps */
        float mean_bias_update = cache_output_bias_updates[i] / NUM_TIME_STEPS;

        /* weight update */
        output_layer_biases[i] += mean_bias_update 
                                + momentum * prev_output_bias_updates[i]
                                + weight_decay_term;

        /* Store updates for momentum calulation in next iteration */
        prev_output_bias_updates[i] = mean_bias_update
                                + momentum * prev_output_bias_updates[i]
                                + weight_decay_term;
    }

    for (int i = 0; i < NUM_RECURRENT_UNITS; i++) {
        for (int j = 0; j < NUM_RECURRENT_UNITS + NUM_INPUTS; j++) {
            /* weight decay */
            float weight_decay_term = -2 * weight_decay * learning_rate *
                    recurrent_layer_weights[i][j];

            float mean_weight_update = cache_recurrent_weight_updates[i][j] / NUM_TIME_STEPS;

            /* summing next layer's weight updates */
            recurrent_layer_weights[i][j] += mean_weight_update
                    + (momentum * prev_recurrent_weight_updates[i][j])
                    + weight_decay_term;

            prev_recurrent_weight_updates[i][j] = mean_weight_update 
                    + (momentum * prev_recurrent_weight_updates[i][j])
                    + weight_decay_term;

        }
        /* weight decay */
        float weight_decay_term = -2 * weight_decay * learning_rate *
                    recurrent_layer_biases[i];

        /* summing next layer's updates */
        float mean_bias_update = cache_recurrent_bias_updates[i] / NUM_TIME_STEPS;
        recurrent_layer_biases[i] += mean_bias_update
                        + (momentum * prev_recurrent_bias_updates[i])
                        + weight_decay_term;

        prev_recurrent_bias_updates[i] = mean_bias_update 
                        + (momentum * prev_recurrent_bias_updates[i])
                        + weight_decay_term;

    }
    
}

void RecurrentNeuralNetworkv2::clear_cache() {

    /* Clear cache of inputs for next pass of network */
    cache_inputs.clear();
    cache_recurrent_units.clear();
//    cache_intermediate_recurrent_outputs.clear();
    cache_outputs.clear();

    /* Clear cache of weights updates */
    for (int i = 0; i < NUM_RECURRENT_UNITS; i++) {
        for (int j = 0; j < NUM_INPUTS + NUM_RECURRENT_UNITS; j++) {
            cache_recurrent_weight_updates[i][j] = 0.0;
        }
        cache_recurrent_bias_updates[i] = 0.0;
    }

    for (int i = 0; i < NUM_OUTPUTS; i++) {
        for (int j = 0; j < NUM_RECURRENT_UNITS; j++) {
            cache_output_weight_updates[i][j] = 0.0;
        }
        cache_output_bias_updates[i] = 0.0;
    }

    /* clear cache of saved gradients */
    cache_recurring_error_term.clear();

    // initialize first instance in cache of 
    // recurrent units to zero
    std::array<float, NUM_RECURRENT_UNITS> zeros;
    for (int i = 0; i < NUM_RECURRENT_UNITS; i++) {
        zeros[i] = 0;
    }
    cache_recurrent_units.push_back(zeros);
}

float RecurrentNeuralNetworkv2::train(
        std::vector<std::array<float, NUM_INPUTS>> &sequence,
        int truth_label) {

    /* Encode the truth label to one-hot */
    std::array<float, NUM_OUTPUTS> one_hot = one_hot_encoding(truth_label);

    // Perform recurrent loop on one data instance
    NUM_TIME_STEPS = sequence.size();
    std::array<float, NUM_OUTPUTS> cumulative_preds;
    for (int i = 0; i < NUM_OUTPUTS; i++) cumulative_preds[i] = 0.0;

    for (int i = 0; i < sequence.size(); i++) {
        std::array<float, NUM_INPUTS> instance = sequence[i];
        std::array<float, NUM_OUTPUTS> pred; // Pred at this time step
        pred = RecurrentNeuralNetworkv2::cell_forward(instance, true);
        for (int k = 0; k < NUM_OUTPUTS; k++) {
            cumulative_preds[k] += pred[k] / NUM_TIME_STEPS;
        }
    }

    /* get mean of error */
    float loss = RecurrentNeuralNetworkv2::cross_entropy_loss(
            cumulative_preds, one_hot);

    /* Perform backpropagation */
    /* First on outer layer */
    std::array<float, NUM_RECURRENT_UNITS> error_grads;
    for (int i = 0; i < NUM_RECURRENT_UNITS; i++) error_grads[i] = 0.0;

    /* Last time step error gradients. Note that this is just used for analysis 
     * purposes, and has no effect on weight updates */
    for (int i = 0; i < NUM_TIME_STEPS; i++) {
        error_grads = RecurrentNeuralNetworkv2::cell_backpropagation(one_hot,
                        cumulative_preds, error_grads, false);
    }

    /* Finally, update weights based on cumulative update vals */
    RecurrentNeuralNetworkv2::perform_weight_updates();

    /* Clear cache for next pass */
    RecurrentNeuralNetworkv2::clear_cache();

    return loss;
}

int RecurrentNeuralNetworkv2::inference(std::vector<std::array<float, NUM_INPUTS>> &sequence) {
    /* Output most likely classification based on current model parameters */
    int num_time_steps = sequence.size();
    std::array<float, NUM_OUTPUTS> cumulative_preds;
    for (int i = 0; i < sequence.size(); i++) {
        std::array<float, NUM_INPUTS> instance = sequence[i];
        std::array<float, NUM_OUTPUTS> preds = RecurrentNeuralNetworkv2::cell_forward(instance);

        for (int i = 0; i < NUM_OUTPUTS; i++) {
            cumulative_preds[i] += preds[i] / num_time_steps;
        }
    }

    // return index of greatest value 
    float largest_val = 0.0;
    int final_pred;
    for (int j = 0; j < NUM_OUTPUTS; j++) {
        if (cumulative_preds[j] > largest_val) {
            largest_val = cumulative_preds[j];
            final_pred = j;
        }
    }

    return final_pred;
}

float RecurrentNeuralNetworkv2::leaky_relu(float x, float alpha) {
    return std::max(alpha * x, x);
}

float RecurrentNeuralNetworkv2::leaky_relu_backprop(float x) {
    /* NOTE: i here is the intermediate output of the layer, 
     * not the activated output */
    if (x > 0) {
        return 1.0;
    }

    else {
        return ALPHA;
    }

}

float RecurrentNeuralNetworkv2::softmax(float x, std::array<float, NUM_OUTPUTS> &x_vector) {
    // Get the max value in the vector 
    float max = x_vector[0];
    for (int i = 1; i < NUM_OUTPUTS; i++) {
        if (max < x_vector[i]) {
            max = x_vector[i];
        }
    }

    // Normalize values of array 
    x = x - max;
    std::array<float, NUM_OUTPUTS> normalized_outputs;
    for (int i = 0; i < NUM_OUTPUTS; i++) {
        normalized_outputs[i] = x_vector[i] - max;
    }

    float denominator = 0.0;
    for (int i = 0; i < NUM_OUTPUTS; i++) {
        denominator += std::exp(normalized_outputs[i]);
    }
//    if (std::isnan(denominator)) {
//        std::cout << "nan\n";
//        std::cout << "nan message\n";
//    }
    float activated_output = ( std::exp(x) / (denominator + OFFSET) );
//    if (std::isnan(activated_output)) {
//        std::cout << "nan\n";
//        std::cout << "nan message\n";
//    }
//    return ( std::exp(x) / (denominator + OFFSET) );
    return activated_output;
}

float RecurrentNeuralNetworkv2::cross_entropy_loss(std::array<float, NUM_OUTPUTS> &preds,
                std::array<float, NUM_OUTPUTS> &targets) {

    float loss = 0.0;
    for (int i = 0; i < NUM_OUTPUTS; i++) {
        loss += -(targets[i] * std::log(preds[i] + OFFSET));
    }

    return loss;
}

std::array<float, NUM_OUTPUTS> RecurrentNeuralNetworkv2::one_hot_encoding(int truth_label) {
    /* Return one hot encoding of label */
    std::array<float, NUM_OUTPUTS> one_hot;
    for (int i = 0; i < NUM_OUTPUTS; i++) {
        one_hot[i] = 0.0;
    }
    one_hot[truth_label] = 1.0;
    return one_hot;
}

/* GRADIENT CLIPPING AND NORMING FUNCTIONS TO TRY AND PREVENT EXPLODING GRADIENTS */
std::array<float, NUM_RECURRENT_UNITS> RecurrentNeuralNetworkv2::gradient_clipping(
        std::array<float, NUM_RECURRENT_UNITS> &gradient) {
    /* Clips gradient values of the network if they exceed a given threshold (see 
     * model hyperparameters) */
    std::array<float, NUM_RECURRENT_UNITS> clipped_gradient;
    for (int i = 0; i < NUM_RECURRENT_UNITS; i++) {
        if (gradient[i] <= -grad_clip_threshold) {
            clipped_gradient[i] = -grad_clip_threshold;
        }
        else if (gradient[i] >= grad_clip_threshold) {
            clipped_gradient[i] = grad_clip_threshold;
        }
        else {
            clipped_gradient[i] = gradient[i];
        }
    }

    return clipped_gradient;
}

std::array<float, NUM_RECURRENT_UNITS> RecurrentNeuralNetworkv2::gradient_norm_scaling(
        std::array<float, NUM_RECURRENT_UNITS> &gradient) {
    /* Applies gradient norm scaling to recurrent gradient terms. This 
     * involves scaling values of gradient vector if the L2 vector norm 
     * exceeds some threshold value */
    float L2_norm = 0.0;
    for (int i = 0; i < NUM_RECURRENT_UNITS; i++) {
        L2_norm += std::pow(gradient[i], 2.0);
    }
    L2_norm = std::sqrt(L2_norm);
    
    std::array<float, NUM_RECURRENT_UNITS> clipped_gradient;
    if (L2_norm >= grad_norm_threshold) {
        for (int i = 0; i < NUM_RECURRENT_UNITS; i++) {
            clipped_gradient[i] = gradient[i] * (grad_norm_threshold / (L2_norm + OFFSET)); 
        }
        return clipped_gradient;
    }
    else {
        return gradient;
    }
}

void RecurrentNeuralNetworkv2::save_weights(std::string filename) {
    /* Save current set of model weights to file */
    std::ofstream ofs(filename, std::ofstream::trunc);

    /* First save recurrent layer weights */
    for (int i = 0; i < NUM_RECURRENT_UNITS; i++) {
        for (int k = 0; k < NUM_INPUTS + NUM_RECURRENT_UNITS; k++) {
            ofs << recurrent_layer_weights[i][k];
            if (k != NUM_INPUTS + NUM_RECURRENT_UNITS - 1) 
                ofs << ",";
        }
        ofs << "\n";
    }
    ofs <<"\n";

    /* Save recurrent biases */
    for (int i = 0; i < NUM_RECURRENT_UNITS; i++) {
        ofs << recurrent_layer_biases[i];
        if (i != NUM_RECURRENT_UNITS - 1) {
            ofs << ",";
        }
        else {
            ofs << "\n\n";
        }
    }

    /* save output layer weights */
    for (int i = 0; i < NUM_OUTPUTS; i++) {
        for (int k = 0; k < NUM_RECURRENT_UNITS; k++) {
            ofs << output_layer_weights[i][k];
            if (k != NUM_RECURRENT_UNITS - 1) 
                ofs << ",";
        }
        ofs << "\n";
    }
    ofs <<"\n";

    /* Save output biases */
    for (int i = 0; i < NUM_OUTPUTS; i++) {
        ofs << output_layer_biases[i];
        if (i != NUM_OUTPUTS - 1)
            ofs << ",";
    }
    ofs << "\n";
}

void RecurrentNeuralNetworkv2::load_weights(std::string filename) {
    /* Loads weights from file. Checks if they are valid first */

    std::ifstream weight_file(filename);
    std::string line;

    recurrent_weights_t loaded_recurrent_weights;
    for (int i = 0; i < NUM_RECURRENT_UNITS; i++) {
        if (std::getline(weight_file, line)) {
            // Create string stream of current line 
            std::stringstream ss(line);
            for (int k = 0; k < NUM_INPUTS + NUM_RECURRENT_UNITS; k++) {
                float val;
                if (ss >> val) {
                    loaded_recurrent_weights[i][k] = val;
                    if (ss.peek() == ',') ss.ignore();
                } else {
                    std::cout << "There was an error loading weights (" << i << "," << k << ") ";
                    std::cout << "for the recurrent layer.\n";
                    std::cout << "Check that weights are appropriate for this ";
                    std::cout << "model architecture\n";
                    return;
                }
            }
        } else {
              std::cout << "There was an error loading weights (" << i << "," << 0 << ") ";
              std::cout << "for the recurrent layer.\n";
              std::cout << "Check that weights are appropriate for this ";
              std::cout << "model architecture\n";
              return;
        }
    }

    // Use getline to skip blank line 
    if (!(std::getline(weight_file, line))) {
        std::cout << "There was an error loading weights ";
        std::cout << "for the recurrent bias.\n";
        std::cout << "Check that weights are appropriate for this ";
        std::cout << "model architecture\n";
        return;
    }

    std::array<float, NUM_RECURRENT_UNITS> loaded_recurrent_bias;
    if (std::getline(weight_file, line)) {

        std::stringstream ss(line);
        for (int i = 0; i < NUM_RECURRENT_UNITS; i++) {
                // Create string stream of current line 
            float val;
            if (ss >> val) {
                loaded_recurrent_bias[i] = val;
                if (ss.peek() == ',') ss.ignore();
            } else {
                std::cout << "There was an error loading weight (" << i << ") ";
                std::cout << "for the recurrent bias.\n";
                std::cout << "Check that weights are appropriate for this ";
                std::cout << "model architecture\n";
                return;
            }
        }
    } else {
         std::cout << "There was an error loading weights ";
         std::cout << "for the recurrent bias.\n";
         std::cout << "Check that weights are appropriate for this ";
         std::cout << "model architecture\n";
         return;
    }

    // Use getline to skip blank line 
    if (!(std::getline(weight_file, line))) {
        std::cout << "There was an error loading weights ";
        std::cout << "for the output layer.\n";
        std::cout << "Check that weights are appropriate for this ";
        std::cout << "model architecture\n";
        return;
    }

    output_weights_t loaded_output_weights;
    for (int i = 0; i < NUM_OUTPUTS; i++) {
        if (std::getline(weight_file, line)) {
            // Create string stream of current line 
            std::stringstream ss(line);
            for (int k = 0; k < NUM_RECURRENT_UNITS; k++) {
                float val;
                if (ss >> val) {
                    loaded_output_weights[i][k] = val;
                    if (ss.peek() == ',') ss.ignore();
                } else {
                    std::cout << "There was an error loading weights (" << i << "," << k << ") ";
                    std::cout << "for the output layer.\n";
                    std::cout << "Check that weights are appropriate for this ";
                    std::cout << "model architecture\n";
                    return;
                }
            }
        } else {
              std::cout << "There was an error loading weights (" << i << "," << 0 << ") ";
              std::cout << "for the recurrent layer.\n";
              std::cout << "Check that weights are appropriate for this ";
              std::cout << "model architecture\n";
              return;
        }
    }

    // Use getline to skip blank line 
    if (!(std::getline(weight_file, line))) {
        std::cout << "There was an error loading weights ";
        std::cout << "for the output bias.\n";
        std::cout << "Check that weights are appropriate for this ";
        std::cout << "model architecture\n";
        return;
    }

    std::array<float, NUM_OUTPUTS> loaded_output_bias;
    if (std::getline(weight_file, line)) {

        std::stringstream ss(line);
        for (int i = 0; i < NUM_OUTPUTS; i++) {
                // Create string stream of current line 
            float val;
            if (ss >> val) {
                loaded_output_bias[i] = val;
                if (ss.peek() == ',') ss.ignore();
            } else {
                std::cout << "There was an error loading weight (" << i << ") ";
                std::cout << "for the recurrent bias.\n";
                std::cout << "Check that weights are appropriate for this ";
                std::cout << "model architecture\n";
                return;
            }
        }
    } else {
         std::cout << "There was an error loading weights ";
         std::cout << "for the recurrent bias.\n";
         std::cout << "Check that weights are appropriate for this ";
         std::cout << "model architecture\n";
         return;
    }

    recurrent_layer_weights = loaded_recurrent_weights;
    recurrent_layer_biases = loaded_recurrent_bias;
    output_layer_weights = loaded_output_weights;
    output_layer_biases = loaded_output_bias;
}

//void print_gradient_to_file(std::vector<std::array<float, NUM_RECURRENT_UNITS>> grad) {

//    std::ofstream outfile("gradients.txt");
//    if (outfile.is_open()) {
//        for (int i = 0; i < grad.size(); i++) {
//            for (int k = 0; k < NUM_RECURRENT_UNITS; k++) {
//                outfile << grad[i][k];
//                if (k != NUM_RECURRENT_UNITS - 1) {
//                    outfile << ",";
//                }
//            }
//            outfile << "\n";
//        }
//    }
//    outfile << "\n";
//}

