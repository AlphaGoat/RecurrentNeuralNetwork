#include <string>
#include <map>
#include <fstream>
#include <dirent.h>
#include "utils.h"

void read_directory(const std::string& name, filename_vector_t& v) {

    DIR* dirp = opendir(name.c_str());
    struct dirent *dp;
    while ((dp = readdir(dirp)) != NULL) {
        v.push_back(name + "/" + dp->d_name);
    }
    closedir(dirp);
}

int find_by_value(std::map<int, std::string> label_map,
                    std::string label) {
    // See if a label is already in the map
    
    // check if map is empty
    if (!label_map.empty()) {

        for (std::pair<int, std::string> entry : label_map) {
            if ( entry.second == label) {
                return entry.first;
            } 
        }
    }
    return -1;
}

void save_value_to_file(std::string filename, 
        float val, bool first_write) {
    /* Save value to file to be read later */
    filename.insert(0, "logs/");
    if (first_write) {
        std::ofstream ofs(filename, std::ofstream::trunc);
        ofs << val;
        ofs << "\n";
    }
    else {
        std::ofstream ofs(filename, std::ofstream::app);
        ofs << val;
        ofs <<"\n";
    }
}


