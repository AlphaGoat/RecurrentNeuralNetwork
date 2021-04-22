#include <string>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <memory>
#include "utils/utils.h"
#include "utils/sign_utils.h"
#include "model/RNNBase.h"
#include "model/MitchellRNNv2.h"
#include "model/RecurrentNeuralNetworkv2.h"

#pragma GCC diagnostic ignored "-Wsign-compare"

#define LABEL_TYPE PERSON_ID

int main(int argc, char **argv) {

    /* Model hyperparameters */
    float learning_rate = 0.01;
    float momentum = 0.01;
    float weight_decay = 0.001;
    int num_epochs = 20;

    // Gradient clipping hyperparameters 
    // (only used for ModifiedRNN. Setting these 
    // for MitchellRNN will do nothing!)
    float gradient_clipping_threshold = 1.0;
    float gradient_norm_threshold = 1.0;
    bool enable_gradient_clipping = false;
    bool enable_gradient_norm_threshold = false;

    bool He_initialization = false;

    int MODEL_FLAG = MITCHELLRNNv2;

    std::string model_tag = "MitchellRNNv2";
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if ((arg == "-l") || (arg == "--learning_rate")) {
            if (i + 1 < argc) {
                std::istringstream ss(argv[i+1]);
                if (!(ss >> learning_rate)) 
                    std::cout << arg << " recieved an invalid input for arg\n";
            }
        }

        else if ((arg == "-n") || (arg == "--num_epochs")) {
            if (i + 1 < argc) {
                std::istringstream ss(argv[i+1]);
                if (!(ss >> num_epochs)) 
                    std::cout << arg << " recieved an invalid input for arg\n";
                
            }
        }

        else if ((arg == "-m") || (arg == "--momentum")) {
            if (i + 1 < argc) {
                std::istringstream ss(argv[i+1]);
                if (!(ss >> momentum)) 
                    std::cout << arg << " recieved an invalid input for arg\n";
                
            }

        }
        else if ((arg == "-d") || (arg == "--weight_decay")) {
            if (i + 1 < argc) {
                std::istringstream ss(argv[i+1]);
                if (!(ss >> weight_decay)) 
                    std::cout << arg << " recieved an invalid input for arg\n";
                
            }
        }
        else if ((arg == "-G") || (arg == "--enable_gradient_clipping")) {
            enable_gradient_clipping = true;
        }

        else if ((arg == "-T") || (arg == "--enable_gradient_norm_threshold")) {
            enable_gradient_norm_threshold = true;
        }

        else if ((arg == "-C") || (arg == "--gradient_clipping_threshold")) {
            if (i + 1 < argc) {
                std::istringstream ss(argv[i+1]);
                if (!(ss >> gradient_clipping_threshold)) {
                    std::cout << arg << " recieved an invalid input for arg " << arg << "\n";
                }
            }
        }
        else if ((arg == "-N") || (arg == "--gradient_norm_threshold")) {
            if (i + 1 < argc) {
                std::istringstream ss(argv[i+1]);
                if (!(ss >> gradient_norm_threshold)) {
                    std::cout << arg << " recieved an invalid input for arg " << arg << "\n";
                }
            }
        }

        else if ((arg == "-H") || (arg == "--He_initialization")) {
            He_initialization = true;
        }


        else if ((arg == "-t") || (arg == "--model_type")) {
            if (i + 1 < argc) {
                std::istringstream ss(argv[i+1]);
                std::string model;
                if (!(ss >> model)) {
                    std::cout << arg << " recieved an invalid input for arg\n";
                    std::cout << "\tPossible Model Types Are: \n";
                    std::cout << "\t\t1. MitchelRNN\n\t\t2. ModifiedRNN\n";
                }

                if (model == "MitchellRNN")  {
                    MODEL_FLAG = MITCHELLRNNv2;
                    model_tag = "MitchennRNNv2";
                }

                else if (model == "ModifiedRNN") {
                    MODEL_FLAG = MODIFIEDRNNv2;
                    model_tag = "ModifiedRNN";
                }

                else {
                    std::cout << arg << " recieved an invalid input for arg\n";
                }
            }
        }
    }


    /* Get test, valid, and test sets */
    filename_vector_t train_dirnames, valid_dirnames, test_dirnames; 
    std::string traindir = "dataset/signs/train";
    std::string validdir = "dataset/signs/valid";
    std::string testdir = "dataset/signs/test";

    // Get sub-directories in base data directory
    read_directory(traindir, train_dirnames);
    read_directory(validdir, valid_dirnames);
    read_directory(testdir, test_dirnames);

    // iterate through sub-directories to get files
    std::vector<std::string> train_filenames, valid_filenames, test_filenames;
    std::vector<int> train_labels, valid_labels, test_labels;
    std::map<int, std::string> label_map;
    for (int i = 0; i < train_dirnames.size(); i++)
        get_data(train_dirnames[i], train_filenames, 
                train_labels, label_map, LABEL_TYPE);

    for (int i = 0; i < valid_dirnames.size(); i++)
        get_data(valid_dirnames[i], valid_filenames,
                valid_labels, label_map, LABEL_TYPE);

    for (int i = 0; i < test_dirnames.size(); i++)
        get_data(test_dirnames[i], test_filenames,
                test_labels, label_map, LABEL_TYPE);


    /* Pair data files with labels */
    std::vector<std::pair<std::string, int>> train_pairs;
    for (int i = 0; i < train_filenames.size(); i++) {
        std::pair<std::string, int> instance;
        instance.first = train_filenames[i];
        instance.second = train_labels[i];
        train_pairs.push_back(instance);
    }

    std::vector<std::pair<std::string, int>> valid_pairs;
    for (int i = 0; i < valid_filenames.size(); i++) {
        std::pair<std::string, int> instance;
        instance.first = valid_filenames[i];
        instance.second = valid_labels[i];
        valid_pairs.push_back(instance);
    }

    std::vector<std::pair<std::string, int>> test_pairs;
    for (int i = 0; i < test_filenames.size(); i++) {
        std::pair<std::string, int> instance;
        instance.first = test_filenames[i];
        instance.second = test_labels[i];
        test_pairs.push_back(instance);
    }

    // initialize recurrent neural network   
    std::unique_ptr<RNNBase> rnn_ptr;
    if (MODEL_FLAG == MODIFIEDRNNv2) {
        rnn_ptr = std::make_unique<RecurrentNeuralNetworkv2>(learning_rate,
                                                             momentum, 
                                                             weight_decay, 
                                                             gradient_clipping_threshold,
                                                             gradient_norm_threshold, 
                                                             enable_gradient_clipping,
                                                             enable_gradient_norm_threshold,
                                                             He_initialization);
    }
    else {
        rnn_ptr = std::make_unique<MitchellRNNv2>(learning_rate,
                                momentum, weight_decay);
    }

    // File to save best epoch's weights to
    std::string weight_file = "logs/signs_";
    weight_file.append(model_tag);
    weight_file.append("_best_weights.txt");
 
    // begin training loop
    float best_validation_loss = 1e5; /* Track best validation loss over 
                                         epochs to decide which set of 
                                         model weights to save */
    for (int n = 0; n < num_epochs; n++) {
        // Randomly shuffle pairs 
        std::random_shuffle(train_pairs.begin(), train_pairs.end(),
                [&] (int i) {
                return std::rand() % i;
                });

        float training_error = 0.0;
        for (int j = 0; j < train_pairs.size(); j++) {

            /* split filename from label */
            std::string filename = train_pairs[j].first;
            int truth_label = train_pairs[j].second;

            // load data instance from file
           sign_data_instance_t data_instance;

            try {
                 data_instance = read_csv(filename);
            } catch (std::runtime_error e1) {
                continue;
            }

            // Apply median filter to data instance 
            data_instance = median_filter_1D(data_instance);


            // PRINT gradient generated by first instance in an epoch to file
            float loss;
//            if (j == train_pairs.size() - 1) {
//                loss = rnn_ptr->train(data_instance, truth_label, true, true);
//            }
//            else {
                loss = rnn_ptr->train(data_instance, truth_label);  
//            }
            training_error += loss; 
        }

        std::cout << "Training error epoch " << n << ": " << training_error << "\n";
        std::string savefile = "_train_error_trainSigns.txt";
        savefile.insert(0, model_tag);
        if (n == 0) {
            save_value_to_file(savefile, training_error, true);
        }
        else {
            save_value_to_file(savefile, training_error, false);
        }

        // Validation loop 
        std::random_shuffle(valid_pairs.begin(), valid_pairs.end(),
                [&] (int i) {
                return std::rand() % i;
                });
        float validation_error = 0.0;
        for (int j = 0; j < valid_pairs.size(); j++) {
            /* grab data pair for this instance */
            std::pair <std::string, int> instance_pair = valid_pairs[j];

            /* split filename from label */
            std::string filename = instance_pair.first;
            int truth_label = instance_pair.second;

            // load data instance from file
//            vowel_data_instance_t data_instance = read_vowel_csv(filename); 
            sign_data_instance_t data_instance = read_csv(filename);

            // Apply median filter to data instance 
            data_instance = median_filter_1D(data_instance);


            // PRINT gradient generated by first instance in an epoch to file
            float loss;
//            if (j == train_pairs.size() - 1) {
//                loss = rnn_ptr->train(data_instance, truth_label, true);
//            }
//            else {
                loss = rnn_ptr->train(data_instance, truth_label);  
//            }
            validation_error += loss; 

        }

        std::cout << "Validation error epoch " << n << ": " << validation_error << "\n";
        std::string val_savefile = "_val_error_trainSigns.txt";
        val_savefile.insert(0, model_tag);
        if (n == 0) {
            save_value_to_file(val_savefile, validation_error, true);
        }
        else {
            save_value_to_file(val_savefile, validation_error, false);
        }
        if (validation_error < best_validation_loss) {
            best_validation_loss = validation_error;
            rnn_ptr->save_weights(weight_file);
        }
    }

    // Load weights from best epoch (minimum validation loss)
    rnn_ptr->load_weights(weight_file);

    ////////////////// RUNNING ON TRAINING SET //////////////////
    ///////////////// FOR TESTING PURPOSES ////////////////////

    // Do final test loop and output model accuracy 
    std::random_shuffle(train_pairs.begin(), train_pairs.end(),
            [&] (int i) {
            return std::rand() % i;
            });

    int num_train = train_pairs.size();
    int num_train_correct = 0;
    for (int j = 0; j < train_pairs.size(); j++) {
        /* grab data pair for this instance */
        std::pair <std::string, int> instance_pair = train_pairs[j];

        /* split filename from label */
        std::string filename;
        try {
            filename = instance_pair.first;
        } catch (std::logic_error const &expr) {
            continue;
        }
        int truth_label = instance_pair.second;

        // load data instance from file
//        vowel_data_instance_t data_instance = read_vowel_csv(filename); 
            sign_data_instance_t data_instance;
            try {
                data_instance = read_csv(filename);
            } catch (std::runtime_error const &expr) {
                std::cout << "Could not open filename: " << filename << "\n";
                continue;
            }

        // Apply median filter to data instance 
//        data_instance = median_filter_1D(data_instance);

        int pred = rnn_ptr->inference(data_instance);
        if (truth_label == pred) {
            num_train_correct++;
        }
    }

    std::cout << "Final Train Accuracy: " << ((float) num_train_correct / (float) (num_train)) << "\n";

    ////////////////// RUNNING ON TRAINING SET //////////////////
    ///////////////// FOR TESTING PURPOSES ////////////////////

    // Do final test loop and output model accuracy 
    std::random_shuffle(test_pairs.begin(), test_pairs.end(),
            [&] (int i) {
            return std::rand() % i;
            });

    // Load best epoch weights 
    rnn_ptr->load_weights(weight_file);

    int num_test = test_pairs.size();
    int num_correct = 0;
    for (int j = 0; j < test_pairs.size(); j++) {
        /* grab data pair for this instance */
        std::pair <std::string, int> instance_pair = test_pairs[j];

        /* split filename from label */
        std::string filename = instance_pair.first;
        int truth_label = instance_pair.second;

        // load data instance from file
        sign_data_instance_t data_instance = read_csv(filename);

        // Apply median filter to data instance 
        data_instance = median_filter_1D(data_instance);

        int pred = rnn_ptr->inference(data_instance);
        if (truth_label == pred) {
            num_correct++;
        }
    }

    std::cout << "Final Test Accuracy: " << ((float) num_correct / (float) (num_test)) << "\n";

}
