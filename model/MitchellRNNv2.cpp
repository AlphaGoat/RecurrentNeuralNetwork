#include "RNNBase.h"
#include "MitchellRNNv2.h"
#include <sstream>
#include <iostream>
#include <fstream>

#define OFFSET 1e-4

#pragma GCC diagnostic ignored "-Wsign-compare"

//std::default_random_engine mitchell_generator;
//std::uniform_real_distribution<float> mitchell_distribution(-1, 1);

/* Generators for random weight initialization */
std::default_random_engine mitchell_generator;
std::uniform_real_distribution<float> mitchell_distribution(-1, 1);


MitchellRNNv2::MitchellRNNv2(float learning_rate, float momentum, float weight_decay,
        bool normal_weight_updates) {
    MitchellRNNv2::learning_rate = learning_rate;
    MitchellRNNv2::momentum = momentum;
    MitchellRNNv2::weight_decay = weight_decay;
    MitchellRNNv2::initialize_weights();
    MitchellRNNv2::NORMAL_UPDATES_FLAG = normal_weight_updates;
}

void MitchellRNNv2::initialize_weights() {
    /* Initialize weights of network to uniform distribution 
     * in range [-1.0, 1.0] */

    srand( (unsigned int) time(NULL));

    // Initialize Recurrent weights 
    for (int i = 0; i < NUM_RECURRENT_UNITS; i++) {
        for (int j = 0; j < NUM_INPUTS + NUM_RECURRENT_UNITS; j++) {
            recurrent_layer_weights[i][j] = mitchell_distribution(mitchell_generator);
        }

        recurrent_layer_biases[i] = mitchell_distribution(mitchell_generator);
    }

    // Initialize output weights
    for (int i = 0; i < NUM_OUTPUTS; i++) {
        for (int j = 0; j < NUM_RECURRENT_UNITS; j++) {
            output_layer_weights[i][j] = mitchell_distribution(mitchell_generator);
        }

        output_layer_biases[i] = mitchell_distribution(mitchell_generator);
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

std::array<float, NUM_OUTPUTS> MitchellRNNv2::cell_forward(
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
    for (int i = 0; i < NUM_RECURRENT_UNITS; i++) {
        float recurrent_i = 0.0;
        for (int j = 0; j < NUM_INPUTS + NUM_RECURRENT_UNITS; j++) {
            recurrent_i += recurrent_layer_weights[i][j] * concat_inputs[j]; 
        }

        recurrent_i += recurrent_layer_biases[i];

        // sigmoid activation 
        current_state[i] = MitchellRNNv2::sigmoid(recurrent_i);
    }
    
    std::array<float, NUM_OUTPUTS> outputs;
    for (int i = 0; i < NUM_OUTPUTS; i++) {
        float output_i = 0.0;
        for (int j = 0; j < NUM_RECURRENT_UNITS; j++) {
            output_i += output_layer_weights[i][j] * current_state[j];
        }

        output_i += output_layer_biases[i];

        // sigmoid activation 
        outputs[i] = MitchellRNNv2::sigmoid(output_i);
    }

    // Cache intermediate layer ouputs 
    if (train == true) {
        cache_inputs.push_back(inputs);
        cache_recurrent_units.push_back(current_state);
        cache_outputs.push_back(outputs);
    }

    // return outputs 
    return outputs;
}

std::array<float, NUM_RECURRENT_UNITS> MitchellRNNv2::cell_backpropagation(
        std::array<float, NUM_OUTPUTS> &targets,
        std::array<float, NUM_OUTPUTS> &outputs,
        std::array<float, NUM_RECURRENT_UNITS> &next_cell_grad,
        bool last_time_step) { // for analysis purposes
    // Perform backpropagation and cache weight updates  

    /* Retrieve layer cache and pop (except for prev state, which we need 
     * in next layer */    
    std::array<float, NUM_OUTPUTS> layer_output = cache_outputs.back();
    std::array<float, NUM_RECURRENT_UNITS> current_state = cache_recurrent_units.back();
    cache_outputs.pop_back();
    cache_recurrent_units.pop_back();

    std::array<float, NUM_INPUTS> input = cache_inputs.back();
    std::array<float, NUM_RECURRENT_UNITS> prev_state = cache_recurrent_units.back();
    cache_inputs.pop_back();

    /* Calculate output error term for this time step */
    /* (error term is the negative gradient of the loss with 
     * respect to the output of the output layer */
    std::array<float, NUM_OUTPUTS> output_error_terms;
    for (int j = 0; j < NUM_OUTPUTS; j++)
        output_error_terms[j] = layer_output[j] * (1 - layer_output[j]) 
                            * (targets[j] - outputs[j]);

    /* Perform updates for output layer */
    for (int j = 0; j < NUM_OUTPUTS; j++) {
        for (int k = 0; k < NUM_RECURRENT_UNITS; k++) {
            cache_output_weight_updates[j][k] += learning_rate * output_error_terms[j]
                            * current_state[k];
        }

        cache_output_bias_updates[j] += learning_rate * output_error_terms[j];
    }

    /* Calculate recurrent error term for this time step */
    /* NOTE: the recurrent error term for this time step is 
     * dependent on the recurrent error terms of ALL previous time steps */
    std::array<float, NUM_RECURRENT_UNITS> current_state_error_grads; // error gradients for this cell's outputs
//    std::array<float, NUM_RECURRENT_UNITS> recurrent_error_grads;  // error gradients for subsequence cell's outputs

//    std::array<float, NUM_RECURRENT_UNITS> last_step_grad = cache_last_step_error_grad.back();
    for (int i = 0; i < NUM_RECURRENT_UNITS; i++) {
        float curr_error_term = 0.0;
        float recur_error_term = 0.0;

        for (int j = 0; j < NUM_OUTPUTS; j++) {
            curr_error_term += output_error_terms[j] * output_layer_weights[j][i];
        }

        for (int k = 0; k < NUM_RECURRENT_UNITS; k++) {
            recur_error_term += next_cell_grad[k] * recurrent_layer_weights[k][i];
        }

        current_state_error_grads[i] = (current_state[i] * (1 - current_state[i]))
                        * (curr_error_term + recur_error_term);

        /* This is just for analysis of the gradient later. Compute each time step's 
         * contribution to the error gradient of the last time step */
//        if (last_time_step) 
//            last_step_grad[i] = current_state_error_grads[i];
//        else
//            last_step_grad[i] = (current_state[i] * (1 - current_state[i])) * last_step_grad[i];
    }

    /* Push the last error grad to cache */
//    cache_last_step_error_grad.push_back(last_step_grad);

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

void MitchellRNNv2::perform_weight_updates() {

    // Get the number of time steps in the sequence
//    int num_time_steps = cache_inputs.size();

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

        }
        /* weight decay */
        float weight_decay_term = -2 * weight_decay * learning_rate *
                    recurrent_layer_biases[i];

        /* summing next layer's updates */
        float mean_bias_update = cache_recurrent_bias_updates[i] / NUM_TIME_STEPS;
        recurrent_layer_biases[i] += mean_bias_update
                        + (momentum * prev_recurrent_bias_updates[i])
                        + weight_decay_term;

    }
    
}

void MitchellRNNv2::clear_cache() {

    /* Clear cache of inputs for next pass of network */
    cache_inputs.clear();
    cache_recurrent_units.clear();
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
    cache_last_step_error_grad.clear();

    // initialize first instance in cache of 
    // recurrent units to zero
    std::array<float, NUM_RECURRENT_UNITS> zeros;
    for (int i = 0; i < NUM_RECURRENT_UNITS; i++) {
        zeros[i] = 0;
    }
    cache_recurrent_units.push_back(zeros);
}

float MitchellRNNv2::train(std::vector<std::array<float, NUM_INPUTS>> &sequence,
        int truth_label) {

    /* Encode truth label using one-of-n encoding */
    std::array<float, NUM_OUTPUTS> encoded_label = one_of_n_encoding(truth_label);

    // Perform recurrent loop on one data instance
    float mse = 0.0;
    NUM_TIME_STEPS = sequence.size();
    std::array<float, NUM_OUTPUTS> cumulative_preds;
    for (int i = 0; i < NUM_OUTPUTS; i++) 
        cumulative_preds[i] = 0.0;

    for (int i = 0; i < sequence.size(); i++) {
        std::array<float, NUM_INPUTS> instance = sequence[i];
        std::array<float, NUM_OUTPUTS> pred; // Pred at this time step
        pred = MitchellRNNv2::cell_forward(instance, true);
        for (int k = 0; k < NUM_OUTPUTS; k++) {
            cumulative_preds[k] += pred[k] / NUM_TIME_STEPS;
        }
//        mse += MitchellRNNv2::mean_square_loss(preds, truth_labels);
    }

    /* get mean of error */
    mse = MitchellRNNv2::mean_square_loss(cumulative_preds, encoded_label);

    /* Perform backpropagation */
    /* First on outer layer */
    std::array<float, NUM_RECURRENT_UNITS> error_grads;
    for (int i = 0; i < NUM_RECURRENT_UNITS; i++) error_grads[i] = 0.0;

    /* Last time step error gradients. Note that this is just used for analysis 
     * purposes, and has no effect on weight updates */

    // Normal backpropagation loop
    for (int i = 0; i < NUM_TIME_STEPS; i++) {
        error_grads = MitchellRNNv2::cell_backpropagation(encoded_label,
                                cumulative_preds, error_grads, false);
    }

    /* Finally, update weights based on cumulative update vals */
    MitchellRNNv2::perform_weight_updates();

    /* Clear cache for next pass */
    MitchellRNNv2::clear_cache();

    return mse;
}

int MitchellRNNv2::inference(std::vector<std::array<float, NUM_INPUTS>> &sequence) {
    /* Output most likely classification based on current model parameters */
    int num_time_steps = sequence.size();
    std::array<float, NUM_OUTPUTS> cumulative_preds;
    for (int i = 0; i < sequence.size(); i++) {
        std::array<float, NUM_INPUTS> instance = sequence[i];
        std::array<float, NUM_OUTPUTS> preds = MitchellRNNv2::cell_forward(instance);

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

float MitchellRNNv2::mean_square_loss(
        /* Returns error value as well as error terms for 
         * mean square error loss */
        std::array<float, NUM_OUTPUTS> &preds,
        std::array<float, NUM_OUTPUTS> &targets) {

    float error = 0.0;
    for (int i = 0; i < NUM_OUTPUTS; i++) {
        error += pow(targets[i] - preds[i], 2.0);
    }
    return (1.0 / 2.0) * error;
}

float MitchellRNNv2::sigmoid(float x) {
    return (1 / (1 + exp(-x)));
}

std::array<float, NUM_OUTPUTS> one_of_n_encoding(int truth_label) {
    /* Encode the label with one-of-n-encoding */
    std::array<float, NUM_OUTPUTS> encoding_vector;
    for (int j = 0; j < NUM_OUTPUTS; j++) {
        if (j == truth_label) {
            encoding_vector[j] = 0.9;
        }
        else {
            encoding_vector[j] = 0.1;
        }
    }
    return encoding_vector;
}

//void print_gradient_to_file(std::vector<std::array<float, NUM_RECURRENT_UNITS>> grad) {
//
//    std::ofstream outfile("gradients.txt");
//    if (outfile.is_open()) {
//        for (int i = 0; i < grad.size(); i++) {
//            for (int k = 0; k < NUM_RECURRENT_UNITS; k++) {
//                outfile << grad[i][k] << " ";
//            }
//            outfile << "\n";
//        }
//    }
//    outfile << "\n";
//}

void MitchellRNNv2::save_weights(std::string filename) {
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
        if (i != NUM_RECURRENT_UNITS - 1)
            ofs << ",";
    }
    ofs << "\n\n";

    /* save output layer weights */
    for (int i = 0; i < NUM_OUTPUTS; i++) {
        for (int k = 0; k < NUM_RECURRENT_UNITS; k++) {
            ofs << output_layer_weights[i][k];
            if (k != NUM_RECURRENT_UNITS - 1) 
                ofs << ",";
        }
        ofs << "\n";
    }
    ofs <<"\n\n";

    /* Save output biases */
    for (int i = 0; i < NUM_OUTPUTS; i++) {
        ofs << output_layer_biases[i];
        if (i != NUM_OUTPUTS - 1)
            ofs << ",";
    }
    ofs << "\n";
}

void MitchellRNNv2::load_weights(std::string filename) {
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

        for (int i = 0; i < NUM_RECURRENT_UNITS; i++) {
                // Create string stream of current line 
                std::stringstream ss(line);
                for (int k = 0; k < NUM_INPUTS + NUM_RECURRENT_UNITS; k++) {
                    float val;
                    if (ss >> val) {
                        loaded_recurrent_bias[i] = val;
                    } else {
                        std::cout << "There was an error loading weight (" << i << ") ";
                        std::cout << "for the recurrent layer.\n";
                        std::cout << "Check that weights are appropriate for this ";
                        std::cout << "model architecture\n";
                        return;
                    }
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
                    loaded_recurrent_weights[i][k] = val;
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

        for (int i = 0; i < NUM_RECURRENT_UNITS; i++) {
                // Create string stream of current line 
                std::stringstream ss(line);
                for (int k = 0; k < NUM_INPUTS + NUM_RECURRENT_UNITS; k++) {
                    float val;
                    if (ss >> val) {
                        loaded_recurrent_bias[i] = val;
                    } else {
                        std::cout << "There was an error loading weight (" << i << ") ";
                        std::cout << "for the output bias.\n";
                        std::cout << "Check that weights are appropriate for this ";
                        std::cout << "model architecture\n";
                        return;
                    }
                }
        }
    } else {
         std::cout << "There was an error loading weights ";
         std::cout << "for the output bias.\n";
         std::cout << "Check that weights are appropriate for this ";
         std::cout << "model architecture\n";
         return;
    }

    recurrent_layer_weights = loaded_recurrent_weights;
    recurrent_layer_biases = loaded_recurrent_bias;
    output_layer_weights = loaded_output_weights;
    output_layer_biases = loaded_output_bias;
}

