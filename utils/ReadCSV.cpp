/* CSV Reader function 
 * solution provided by Ben Gorman on GormAnalysis
 * https://www.gormanalysis.com/blog/reading-and-writing-csv-files-with-cpp/
 */
#include <string>
#include <fstream>
#include <array>
#include <vector>
#include <utility> // std::pair
#include <stdexcept> // std::runtime_error
#include <sstream> // std::stringstream
//#include "vowel_utils.h"
//#include "sign_utils.h"
#include "utils.h"

#ifndef NUM_INPUTS 
#define NUM_INPUTS 12 
#endif

#pragma GCC diagnostic ignored "-Wsign-compare"

std::vector<std::array<float, NUM_INPUTS>> read_csv(const std::string &filename) {
    // Reads a CSV file into a vector of <string, vector<int>> pairs where
    // each pair represents <column name, column values>

    // Create a vector of <string, int vector> pairs to store the result
    std::vector<std::array<float, NUM_INPUTS>> data_instance;

    // Create an input filestream
    std::ifstream myFile(filename);

    // Make sure the file is open
    if(!myFile.is_open()) throw std::runtime_error("Could not open file");

    // Helper vars
    std::string line, colname;
    float val;

    // Read data, line by line
    while(std::getline(myFile, line))
    {
        // Create a stringstream of the current line
        std::stringstream ss(line);

        // Initialize array to hold row values
        std::array<float, NUM_INPUTS> row_values;
        
        // Keep track of the current column index
        int colIdx = 0;
        
        // Extract each integer
        while(ss >> val && colIdx <= NUM_INPUTS - 1){
            
            // Add the current integer to the 'colIdx' column's values vector
//            result.at(colIdx).second.push_back(val);
            row_values[colIdx] = val;
            
            // If the next token is a comma, ignore it and move on
            if(ss.peek() == ',') ss.ignore();
            
            // Increment the column index
            colIdx++;
        }

        data_instance.push_back(row_values);
    }

    // Close file
    myFile.close();

    return data_instance;
}

//sign_data_instance_t read_sign_csv(const std::string &filename) {
//    // Reads a CSV file into a vector of <string, vector<int>> pairs where
//    // each pair represents <column name, column values>
//
//    // Create a vector of <string, int vector> pairs to store the result
//    sign_data_instance_t data_instance;
//
//    // Create an input filestream
//    std::ifstream myFile(filename);
//
//    // Make sure the file is open
//    if(!myFile.is_open()) throw std::runtime_error("Could not open file");
//
//    // Helper vars
//    std::string line, colname;
//    float val;
//
//    // Read data, line by line
//    while(std::getline(myFile, line))
//    {
//        // Create a stringstream of the current line
//        std::stringstream ss(line);
//
//        // Initialize array to hold row values
//        std::array<float, 10> row_values;
//        
//        // Keep track of the current column index
//        int colIdx = 0;
//        
//        // Extract each integer
//        while(ss >> val && colIdx <= 9){
//            
//            // Add the current integer to the 'colIdx' column's values vector
////            result.at(colIdx).second.push_back(val);
//            row_values[colIdx] = val;
//            
//            // If the next token is a comma, ignore it and move on
//            if(ss.peek() == ',') ss.ignore();
//            
//            // Increment the column index
//            colIdx++;
//        }
//
//        data_instance.push_back(row_values);
//    }
//
//    // Close file
//    myFile.close();
//
//    return data_instance;
//}
//
//vowel_data_instance_t read_vowel_csv(const std::string &filename) {
//    // Reads a CSV file into a vector of <string, vector<int>> pairs where
//    // each pair represents <column name, column values>
//
//    // Create a vector of <string, int vector> pairs to store the result
//    vowel_data_instance_t data_instance;
//
//    // Create an input filestream
//    std::ifstream myFile(filename);
//
//    // Make sure the file is open
//    if(!myFile.is_open()) throw std::runtime_error("Could not open file");
//
//    // Helper vars
//    std::string line, colname;
//    float val;
//
//    // Read data, line by line
//    while(std::getline(myFile, line))
//    {
//        // Create a stringstream of the current line
//        std::stringstream ss(line);
//
//        // Initialize array to hold row values
//        std::array<float, 12> row_values;
//        
//        // Keep track of the current column index
//        int colIdx = 0;
//        
//        // Extract each integer
//        while(ss >> val && colIdx <= 11){
//            
//            // Add the current integer to the 'colIdx' column's values vector
////            result.at(colIdx).second.push_back(val);
//            row_values[colIdx] = val;
//            
//            // If the next token is a comma, ignore it and move on
//            if(ss.peek() == ',') ss.ignore();
//            
//            // Increment the column index
//            colIdx++;
//        }
//
//        data_instance.push_back(row_values);
//    }
//
//    // Close file
//    myFile.close();
//
//    return data_instance;
//}
//
std::vector<std::array<float, 5>> read_iris_csv(const std::string &filename) {
    // Reads a CSV file into a vector of <string, vector<int>> pairs where
    // each pair represents <column name, column values>

    // Create a vector of <string, int vector> pairs to store the result
    std::vector<std::array<float, 5>> data_instance;

    // Create an input filestream
    std::ifstream myFile(filename);

    // Make sure the file is open
    if(!myFile.is_open()) throw std::runtime_error("Could not open file");

    // Helper vars
    std::string line, colname;
    float val;

    // Read data, line by line
    while(std::getline(myFile, line))
    {
        // Create a stringstream of the current line
        std::stringstream ss(line);

        // Initialize array to hold row values
        std::array<float, 5> row_values;
        
        // Keep track of the current column index
        int colIdx = 0;
        
        // Extract each integer
        while(ss >> val && colIdx <= 5){
            
            // Add the current integer to the 'colIdx' column's values vector
//            result.at(colIdx).second.push_back(val);
            row_values[colIdx] = val;
            
            // If the next token is a comma, ignore it and move on
            if(ss.peek() == ',') ss.ignore();
            
            // Increment the column index
            colIdx++;
        }

        data_instance.push_back(row_values);
    }

    // Close file
    myFile.close();

    return data_instance;
}

//////////////////////////
// FOR TESTING PURPOSES//
//////////////////////////
//void write_sign_csv(const std::string &filename, const sign_data_instance_t &dataset){
//    // Make a CSV file with one or more columns of integer values
//    // Each column of data is represented by the pair <column name, column data>
//    //   as std::pair<std::string, std::vector<int>>
//    // The dataset is represented as a vector of these columns
//    // Note that all columns should be the same size
//    
//    // Create an output filestream object
//    std::ofstream myFile(filename);
//    
//    
//    // Send data to the stream
//    for(int i = 0; i < dataset.size(); ++i)
//    {
//        std::array<float, 10> vals = dataset[i];
//        for(int j = 0; j < dataset[i].size(); ++j)
//        {
//            myFile << vals[j];
//            if(j != dataset.size() - 1) myFile << ","; // No comma at end of line
//        }
//        myFile << "\n";
//    }
//    
//    // Close the file
//    myFile.close();
//}

void write_csv(const std::string &filename, std::vector<std::array<float, NUM_INPUTS>> &dataset){
    // Make a CSV file with one or more columns of integer values
    // Each column of data is represented by the pair <column name, column data>
    //   as std::pair<std::string, std::vector<int>>
    // The dataset is represented as a vector of these columns
    // Note that all columns should be the same size
    
    // Create an output filestream object
    std::ofstream myFile(filename);
    
    // Send data to the stream
    for(int i = 0; i < dataset.size(); ++i)
    {
        std::array<float, NUM_INPUTS> vals = dataset[i];
        for(int j = 0; j < dataset[i].size(); ++j)
        {
            myFile << vals[j];
            if(j != dataset.size() - 1) myFile << ","; // No comma at end of line
        }
        myFile << "\n";
    }
    
    // Close the file
    myFile.close();
}
