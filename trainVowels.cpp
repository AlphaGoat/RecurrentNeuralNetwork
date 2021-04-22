#include <string>
#include <utility>
#include <iostream>
#include <fstream>
#include <sstream>
#include <memory>
#include <algorithm>
#include "utils/utils.h"
#include "utils/vowel_utils.h"
#include "model/RNNBase.h"
#include "model/MitchellRNNv2.h"
#include "model/RecurrentNeuralNetworkv2.h"

#pragma GCC diagnostic ignored "-Wsign-compare"

int main(int argc, char **argv) {

    /* Model hyperparameters */
    float learning_rate = 0.01;
    float momentum = 0.01;
    float weight_decay = 0.001;
    int num_epochs = 20;

    /* Gradient clipping hyperparameters */
    float gradient_clipping_threshold = 1.0;
    float gradient_norm_threshold = 1.0;

    /* Flags to set to enable different gradient flow 
     * and intitialization schema */
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
                    std::cout << arg << " recieved an invalid input for arg " << arg << "\n";
            }
        }

        else if ((arg == "-n") || (arg == "--num_epochs")) {
            if (i + 1 < argc) {
                std::istringstream ss(argv[i+1]);
                if (!(ss >> num_epochs)) 
                    std::cout << arg << " recieved an invalid input for arg " << arg << "\n";
                
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
                if (!(ss >> weight_decay)) 
                    std::cout << arg << " recieved an invalid input for arg " << arg << "\n";
                
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
                if (!(ss >> model))
                    std::cout << arg << " recieved an invalid input for arg\n";

                if (model == "MitchellRNN") {
                    MODEL_FLAG = MITCHELLRNNv2;
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

    ////////////// FETCH TRAIN DATA ////////////////////
    filename_vector_t train_dirnames; 
    std::string train_base_dirname = "dataset/vowels/train";

    // Get sub-directories in base data directory
    read_directory(train_base_dirname, train_dirnames);

    // iterate through sub-directories to get files
    std::vector<std::string> train_filenames;
    std::vector<int> train_labels;
    std::map<int, std::string> train_label_map;
    for (int i = 0; i < train_dirnames.size(); i++) {
        get_vowel_data(train_dirnames[i], train_filenames, train_labels, 
                train_label_map);
    }

    // pair data files with labels
    std::vector<std::pair<std::string, int>> train_pairs;
    for (int i = 0; i < train_filenames.size(); i++) {
        std::pair<std::string, int> instance;
        instance.first = train_filenames[i];
        instance.second = train_labels[i];
        train_pairs.push_back(instance);
    }
    ////////////// FETCH TRAIN DATA ////////////////////

    ////////////// FETCH VALID DATA ////////////////////
    filename_vector_t valid_dirnames; 
    std::string valid_base_dirname = "dataset/vowels/val";

    // Get sub-directories in base data directory
    read_directory(valid_base_dirname, valid_dirnames);

    // iterate through sub-directories to get files
    std::vector<std::string> valid_filenames;
    std::vector<int> valid_labels;
    std::map<int, std::string> valid_label_map;
    for (int i = 0; i < valid_dirnames.size(); i++) {
        get_vowel_data(valid_dirnames[i], valid_filenames, valid_labels, 
                valid_label_map);
    }

    // pair data files with labels
    std::vector<std::pair<std::string, int>> valid_pairs;
    for (int i = 0; i < valid_filenames.size(); i++) {
        std::pair<std::string, int> instance;
        instance.first = valid_filenames[i];
        instance.second = valid_labels[i];
        valid_pairs.push_back(instance);
    }
    ////////////// FETCH VALID DATA ////////////////////
    
    ////////////// FETCH TEST DATA ////////////////////
    filename_vector_t test_dirnames; 
    std::string test_base_dirname = "dataset/vowels/test";

    // Get sub-directories in base data directory
    read_directory(test_base_dirname, test_dirnames);

    // iterate through sub-directories to get files
    std::vector<std::string> test_filenames;
    std::vector<int> test_labels;
    std::map<int, std::string> test_label_map;
    for (int i = 0; i < test_dirnames.size(); i++) {
        get_vowel_data(test_dirnames[i], test_filenames, test_labels, 
                test_label_map);
    }

    // pair data files with labels
    std::vector<std::pair<std::string, int>> test_pairs;
    for (int i = 0; i < test_filenames.size(); i++) {
        std::pair<std::string, int> instance;
        instance.first = test_filenames[i];
        instance.second = test_labels[i];
        test_pairs.push_back(instance);
    }


    // initialize recurrent neural network
//    MitchellRNNv2 rnn(learning_rate, momentum, weight_decay);     
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

    std::string weight_file = "logs/vowels_";
    weight_file.append(model_tag);
    weight_file.append("_best_weights.txt");

    float best_validation_loss = 1e5;

    // begin training loop
    for (int i = 0; i < num_epochs; i++) {
        // Randomly shuffle pairs
        std::random_shuffle(train_pairs.begin(), train_pairs.end(),
                [&] (int i) {
                return std::rand() % i;
                });
        
        float training_error = 0.0;
        for (int j = 0; j < train_pairs.size(); j++) {

            /* grab data pair for this instance */
            std::pair <std::string, int> instance_pair = train_pairs[j];

            /* split filename from label */
            std::string filename = instance_pair.first;
            int truth_label = instance_pair.second;

            // load data instance from file
//            vowel_data_instance_t data_instance = read_vowel_csv(filename); 
            vowel_data_instance_t data_instance;
            try {
                data_instance = read_csv(filename);
            } catch (std::runtime_error const& ex) {
                std::cout << "Could not open filename: " << filename << "\n";
                continue;
            }

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

        std::cout << "Training error epoch " << i << ": " << training_error << "\n";
        std::string savefile = "_train_error_trainVowels.txt";
        savefile.insert(0, model_tag);
        if (i == 0) {
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
            vowel_data_instance_t data_instance;
            try {
                data_instance = read_csv(filename);
            } catch (std::runtime_error const& ex) {
                std::cout << "Could not open filename: " << filename << "\n";
                continue;
            }

            // PRINT gradient generated by last instance in an epoch to file
            float loss;
//            if (j == train_pairs.size() - 1) {
//                loss = rnn_ptr->train(data_instance, truth_label, true, true);
//            }
//            else {
            loss = rnn_ptr->train(data_instance, truth_label);  
//            }
            validation_error += loss; 

        }

        std::cout << "Validation error epoch " << i << ": " << validation_error << "\n";
        std::string val_savefile = "_val_error_trainVowels.txt";
        val_savefile.insert(0, model_tag);
        if (i == 0) {
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
            vowel_data_instance_t data_instance;
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

    int num_test = test_pairs.size();
    int num_correct = 0;
    for (int j = 0; j < test_pairs.size(); j++) {
        /* grab data pair for this instance */
        std::pair <std::string, int> instance_pair = test_pairs[j];

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
            vowel_data_instance_t data_instance;
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
            num_correct++;
        }
    }

    std::cout << "Final Test Accuracy: " << ((float) num_correct / (float) (num_test)) << "\n";

}
