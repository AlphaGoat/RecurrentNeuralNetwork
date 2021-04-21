#ifndef IRIS_PARAMS_H
#define IRIS_PARAMS_H

#include <vector>
#include <array>


#define NUM_INPUTS 4 
#define NUM_OUTPUTS 3
#define NUM_RECURRENT_UNITS 3

std::vector<std::array<float, 5>> read_csv(const std::string &filename);

#endif
