#include <string>
#include <utility>
#include <iostream>
#include <algorithm>
#include <memory>
#include <sstream>
#include "utils/utils.h"
#include "utils/iris_params.h"
#include "model/MitchellRNNv2.h"
#include "model/RecurrentNeuralNetworkv2.h"

#pragma GCC diagnostic ignored "-Wsign-compare"

//#define NUM_EPOCHS 2500
std::vector<std::array<float, 5>> read_iris_csv(const std::string &filename);

int main(int argc, char **argv) {

    /* hyper parameters */
    float learning_rate = 0.001;
    float momentum = 0.001;
    float weight_decay = 0.001;
    int num_epochs = 2500;
    int MODEL_FLAG = MITCHELLRNNv2;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if ((arg == "-l") || (arg == "--learning_rate")) {
            if (i + 1 < argc) {
                std::istringstream ss(argv[i+1]);
                if (!(ss >> learning_rate)) 
                    learning_rate = atof(argv[i++]);
            }
        }

        else if ((arg == "-n") || (arg == "--num_epochs")) {
            if (i + 1 < argc) {
                std::istringstream ss(argv[i+1]);
                if (!(ss >> num_epochs)) 
                    std::cout << arg << " recieved an invalid input\n";
                
            }
        }

        else if ((arg == "-m") || (arg == "--momentum")) {
            if (i + 1 < argc) {
                std::istringstream ss(argv[i+1]);
                if (!(ss >> momentum)) 
                    std::cout << arg << " recieved an invalid input for arg " << arg << "\n";
                
            }

        }

        else if ((arg == "-d") || (arg == "--weight_decay")) {
            if (i + 1 < argc) {
                std::istringstream ss(argv[i+1]);
                if (!(ss >> num_epochs)) 
                    std::cout << arg << " recieved an invalid input for arg " << arg << "\n";
                
            }
        }
        else if ((arg == "-t") || (arg == "--model_type")) {
            if (i + 1 < argc) {
                std::istringstream ss(argv[i+1]);
                std::string model;
                if (!(ss >> model))
                    std::cout << arg << " recieved an invalid input for arg\n";

                if (model == "MitchellRNN") 
                    MODEL_FLAG = MITCHELLRNNv2;

                else if (model == "ModifiedRNN") 
                    MODEL_FLAG = MODIFIEDRNNv2;

                else {
                    std::cout << arg << " recieved an invalid input for arg\n";
                }
            }
        }
    }

    std::string filename = "dataset/iris/encoded-iris-train.txt";
    std::string test_filename = "dataset/iris/encoded-iris-test.txt";

    std::vector<std::array<float, 5>> data_instances = read_iris_csv(filename); 
    std::vector<std::array<float, 5>> test_instances = read_iris_csv(test_filename); 

    /* separate labels from data */
    std::vector<int> labels;
    std::vector<std::array<float, 4>> attributes;
    std::vector<std::array<float, 4>> test_attributes;
    std::vector<int> test_labels;

    for (int i = 0; i < data_instances.size(); i++) {
        std::array<float, 4> instance_attributes;

        for (int j = 0; j < 4; j++) {
            instance_attributes[j] = data_instances[i][j];
        }

        attributes.push_back(instance_attributes);

        int label = (int) data_instances[i][4];
        labels.push_back(label);

//        std::array<float, 3> one_of_n_encoding;
//        for (int j = 0; j < 3; j++) {
//            if (j == label) {
//                one_of_n_encoding[j] = 0.9;
//            }
//            else {
//                one_of_n_encoding[j] = 0.1;
//            }
//        }
//        labels.push_back(one_of_n_encoding);
        

    }

    for (int i = 0; i < test_instances.size(); i++) {
        std::array<float, 4> instance_attributes;

        for (int j = 0; j < 4; j++) {
            instance_attributes[j] = test_instances[i][j];
        }

        test_attributes.push_back(instance_attributes);

        test_labels.push_back((int) data_instances[i][4]);
    }

    // initialize RNN
//    RNNBase rnn;
//    if (MODEL_FLAG == MITCHELLRNNv2) {
//        rnn = MitchellRNNv2(learning_rate, momentum, weight_decay);
//    }
//    else {
//        rnn = RecurrentNeuralNetworkv2(learning_rate, momentum, weight_decay);
//    }

    std::unique_ptr<RNNBase> rnn_ptr;
 //   try {
    if (MODEL_FLAG == MODIFIEDRNNv2) {
//        auto ptr = std::make_unique<RecurrentNeuralNetworkv2>(learning_rate, 
//                momentum, weight_decay);
//        rnn = new RecurrentNeuralNetworkv2(learning_rate, momentum, weight_decay);
//        rnn_ptr = std::move(ptr);
        rnn_ptr.reset(new RecurrentNeuralNetworkv2(learning_rate, 
                    momentum, weight_decay));
    }
    else {
//        auto ptr = std::make_unique<MitchellRNNv2>(learning_rate,
//                momentum, weight_decay);
//        rnn_ptr = std::move(ptr);
//        rnn = new MitchellRNNv2(learning_rate, momentum, weight_decay);
        rnn_ptr.reset(new MitchellRNNv2(learning_rate,
                    momentum, weight_decay));
    }


    // begin training loop
    for (int i = 0; i < num_epochs; i++) {
        
        float training_error = 0.0;
        for (int j = 0; j < attributes.size(); j++) {

            std::vector<std::array<float, 4>> instance;
            instance.push_back(attributes[j]);

            /* Load data instance from file */
            float error = rnn_ptr->train(instance, labels[j]);  
            training_error += error;
        }

        int num_examples = attributes.size();
        int num_correct = 0;
        for (int j = 0; j < attributes.size(); j++) {

            std::vector<std::array<float, 4>> instance;
            instance.push_back(attributes[j]);

            int truth_label = data_instances[j][4];

            /* Load data instance from file */
//            float error = rnn.train(instance, labels[j]);  
            int pred = rnn_ptr->inference(instance);

//            float largest_val = 0.0;
//            int truth_label;
//            for (int k = 0; k < NUM_OUTPUTS; k++) {
//                if (labels[j][k] > largest_val) {
//                    largest_val = labels[j][k];
//                    truth_label = k;
//                }
//            }
            if (truth_label == pred) {
                num_correct++;
            }

            if (i == 900) {

                int r = 4;
            }
        }

//        float accuracy = ((float) num_correct / (float) num_examples) * 100.0;
        std::cout << "Training error epoch " << i << ": " << training_error;
        std::cout <<"\tnum_correct: " << num_correct << "\n";


        /* validation check */
//        int num_examples = test_attributes.size();
//        int num_correct = 0;
//        for (int k = 0; k < test_attributes.size(); k++) {
//
//            std::vector<std::array<float, 4>> instance;
//            instance.push_back(test_attributes[k]);
//
//            int pred = rnn.inference(instance);
//            
//            if (pred == test_labels[k]) {
//                num_correct++;
//            }
//
//        }
//
//        float test_accuracy = ((float) num_correct / (float) num_examples) * 100.0;
//        std::cout << "Training error epoch " << i << ": " << training_error;
//        std::cout <<"\ttesting accuracy: " << test_accuracy << "\n";
    }
}
