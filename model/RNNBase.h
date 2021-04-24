#ifndef RNNBASE_H
#define RNNBASE_H

#include <array>
#include <cmath>
#include <vector>
#include <iostream>
#include <cstdlib>
#include <ctime>
#include <random>
#include <assert.h>
//#ifdef SIGN_UTILS_H 
//    #include "../utils/sign_utils.h"
//
//#elif defined VOWEL_UTILS_H 
//    #include "../utils/vowel_utils.h"
//#endif

//#ifdef SIGN_UTILS_H
//#define NUM_INPUTS 10
//#define NUM_RECURRENT_UNITS 5
//#define NUM_OUTPUTS 4
//
//#elif defined VOWEL_UTILS_H 
//#define NUM_INPUTS 12 
//#define NUM_RECURRENT_UNITS 6
//#define NUM_OUTPUTS 9
//
//#else 
//#define NUM_INPUTS 10
//#define NUM_RECURRENT_UNITS 5
//#define NUM_OUTPUTS 4
//#endif

#ifndef NUM_INPUTS
#define NUM_INPUTS 12
#endif

#ifndef NUM_RECURRENT_UNITS
#define NUM_RECURRENT_UNITS 6
#endif 

#ifndef NUM_OUTPUTS
#define NUM_OUTPUTS 9
#endif

#define LOSS_FUNCTION_MEAN_SQUARE_ERROR

enum MODEL_TYPE {
    MITCHELLRNNv2,
    MODIFIEDRNNv2
};

//typedef std::vector<std::vector<float>> recurrent_weights_t;
//typedef std::vector<std::vector<float>> output_weights_t;

class RNNBase {

    // Model hyperparameters
    float learning_rate;
    float momentum;
    float weight_decay;


//    recurrent_weights_t recurrent_layer_weights;
//    output_weights_t output_layer_weights;
//
//    std::vector<float> recurrent_layer_biases;
//    std::vector<float> output_layer_biases;
//
//    // Cached cumulative sum of update values 
//    recurrent_weights_t cache_recurrent_weight_updates;
//    output_weights_t cache_output_weight_updates;
//    std::vector<float> cache_recurrent_bias_updates;
//    std::vector<float> cache_output_bias_updates;

    public:
        bool NORMAL_UPDATES_FLAG;

//        RNNBase(float learning_rate, float momentum, float weight_decay);
//        RNNBase(float learning_rate, float momentum, float weight_decay);
        virtual float train(std::vector<std::array<float, NUM_INPUTS>> &sequence,
                int truth_label) = 0;
        virtual int inference(std::vector<std::array<float, NUM_INPUTS>> &sequence) = 0;
        virtual void save_weights(std::string filename) = 0;
        virtual void load_weights(std::string filename) = 0;
        virtual ~RNNBase() {}
        float normal_distribution(float x, float mean, float stddev);

    private:

//    private:

        // Activation functions
//        float sigmoid(float x);
//        float leaky_relu(float x);
//        float softmax(float x, std::vector<float> &x_vector);
//
//        // Loss function
//        float mean_square_loss(std::vector<float> &preds,
//                               std::vector<float> &targets
//                               );
//
//        void initialize_weights();
//        std::array<float, NUM_OUTPUTS> cell_forward(
//                            std::array<float, NUM_INPUTS> &inputs,
//                            bool train = false);
//        std::array<float, NUM_RECURRENT_UNITS> cell_backpropagation(
//                std::array<float, NUM_OUTPUTS> &targets,
//                std::array<float, NUM_OUTPUTS> &preds,
//                std::array<float, NUM_RECURRENT_UNITS> &next_grad,
//                bool last_time_step = false);
//
//        void perform_weight_updates();
//        void clear_cache();


};

void print_gradient_to_file(std::string model_name, 
        std::vector<std::array<float, NUM_RECURRENT_UNITS>> gradient);

#endif
