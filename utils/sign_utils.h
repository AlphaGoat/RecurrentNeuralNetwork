#ifndef SIGN_UTILS_H
#define SIGN_UTILS_H

#include <stdlib.h>
#include <vector>
#include <utility>
#include <map>
#include <string>

/* MODEL PARAMETERS */
#define NUM_INPUTS 10
#define NUM_OUTPUTS 5 
#define NUM_RECURRENT_UNITS 4

/* PARAMETERS FOR DATA UTILITIES */
#define MEDIAN_WINDOW_SIZE 5

struct handsign_data {
    double x;
    double y;
    double z;
    double roll;
    double pitch;
    double yaw;
    double thumb;
    double fore;
    double index;
    double ring;
};

enum LabelType {
    SIGN_ID = 0,
    PERSON_ID = 1
};

//std::map< std::string, int >  LabelAssign;
//std::map< int, std::string >  LabelLookup;

typedef std::vector<std::array<float, NUM_INPUTS>> sign_data_instance_t;
typedef std::vector<std::pair<std::string, std::string>> labeled_data_file_t;


// HELPER FUNCTIONS FOR READING DATA 
//void read_directory(const std::string& name, filename_vector_t& v);
void get_data_by_sign_type(const std::string &datadir, 
        labeled_data_file_t& labeled_data);

void get_data(const std::string &datadir,
        std::vector<std::string> &filenames,
        std::vector<int> &labels,
        std::map<int, std::string> &label_map,
        LabelType label_type_flag);

std::string extract_signer_name(const std::string &dirpath);
std::string extract_sign_type(const std::string &dirpath);

//int find_by_value(std::map<int, std::string> label_map,
//                    std::string label);


sign_data_instance_t read_csv(const std::string &filename);

void write_sign_csv(const std::string &filename, const sign_data_instance_t &dataset);

// DATA PREPROCESSING 
sign_data_instance_t median_filter_1D(sign_data_instance_t &data_instance);

//std::vector < std::string > scroll_directory(const char **dirpath);
//std::vector < handsign_data >(std::string filename);



#endif
