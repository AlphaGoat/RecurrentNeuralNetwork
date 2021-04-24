#include <array>
#include <vector>
#include <fstream>
#include "RNNBase.h"

void print_gradient_to_file(std::string model_name,
        std::vector<std::array<float, NUM_RECURRENT_UNITS>> grad) {

    // Check if values are NaN. If they are, don't print to file
    for (int i = 0; i < grad.size(); i++) {
        for (int k = 0; k < NUM_RECURRENT_UNITS; k++) {
           if (std::isnan(grad[i][k])) {
               return;
           }

        }
    }

    model_name.append("_gradients.txt", std::ofstream::trunc);
    std::ofstream outfile(model_name);
    if (outfile.is_open()) {
        for (int i = 0; i < grad.size(); i++) {
            for (int k = 0; k < NUM_RECURRENT_UNITS; k++) {
                outfile << grad[i][k];
                if (k != NUM_RECURRENT_UNITS - 1) {
                    outfile << ",";
                }
            }
            outfile << "\n";
        }
    }
    outfile << "\n";
}

float RNNBase::normal_distribution(float x, float mean, float stddev) {
    /* Get the probability value of the normal 
     * distribution at some x given a mean and 
     * stddev */
    float expTerm = std::exp((-1.0/2.0) * std::pow((x - mean) / stddev, 2.0));
    return (1.0 / (stddev * std::sqrt(2 * M_PI))) * expTerm;
}
