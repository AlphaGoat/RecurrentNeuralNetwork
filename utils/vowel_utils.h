#ifndef VOWEL_UTILS_H
#define VOWEL_UTILS_H

#include <array>
#include <map>
#include <string>
#include <vector>

#ifndef NUM_INPUTS
#define NUM_INPUTS 12 
#endif

#ifndef NUM_OUTPUTS
#define NUM_OUTPUTS 9
#endif

#ifndef NUM_RECURRENT_UNITS
#define NUM_RECURRENT_UNITS 6
#endif

typedef std::vector<std::array<float, NUM_INPUTS>> vowel_data_instance_t;

void get_vowel_data(const std::string &datadir,
        std::vector<std::string> &filenames,
        std::vector<int> &labels,
        std::map<int, std::string> &label_map);

vowel_data_instance_t read_csv(const std::string &filename);
//vowel_data_instance_t read_vowel_csv(const std::string &filename);

void write_vowel_csv(const std::string &filename, vowel_data_instance_t &data_instance);

#endif
