#ifndef UTILS_H
#define UTILS_H

#include <vector>
#include <string>
#include <map>

typedef std::vector<std::string> filename_vector_t;
void read_directory(const std::string& name, filename_vector_t& v);
int find_by_value(std::map<int, std::string> label_map,
                    std::string label);
void save_value_to_file(std::string filename, float val, bool first_write);
//std::vector<std::array<float, 5>> read_iris_csv(const std::string &filename);
//std::vector<std::array<float, 5>> read_csv(const std::string &filename);

#endif
