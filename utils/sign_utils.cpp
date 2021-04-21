#include <sys/types.h>
#include <dirent.h>
#include <map>
#include <string>
#include <iterator>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include "utils.h"
#include "sign_utils.h"


void get_data_by_sign_type(const std::string &datadir, 
        labeled_data_file_t& labeled_data) {
    // Get data files labeled by the sign being demonstrated 
    DIR* dirp = opendir(datadir.c_str());
    struct dirent *dp;
    while ((dp = readdir(dirp)) != NULL) {
        std::pair<std::string, std::string> datum;

        //label 
        std::string label = dp->d_name;
        size_t i = label.rfind('.', label.length());
        if (i != std::string::npos) {
            label = label.substr(0, i-1);
        }
        datum.first = label; 

        // filename
        datum.second = datadir + "/" + dp->d_name; 
        labeled_data.push_back(datum);
    }
    closedir(dirp);
}

void get_data(const std::string &datadir,
        std::vector<std::string> &filenames,
        std::vector<int> &labels,
        std::map<int, std::string> &label_map,
        LabelType label_type_flag) {

    // Get data files labeled by the identity of the signed
    DIR* dirp = opendir(datadir.c_str());
    struct dirent *dp;
    while ((dp = readdir(dirp)) != NULL) {

        /* Check that the filename isn't just 'this' directory 
         * or 'prev' */
        std::string filename = dp->d_name;
        if (filename == "." || filename == "..") {
            continue;
        }

        std::string::size_type extension = filename.rfind('.');
        if (extension == std::string::npos) {
            continue;
        }

        std::string label_str;
        if (label_type_flag == SIGN_ID) {
            label_str = extract_sign_type(filename);
        }

        else {
            label_str = extract_signer_name(datadir); 
        }

        // Add label to hashmap, if it is not already there.
        // Otherwise, assign an integer label to data if it is
        int key = find_by_value(label_map, label_str);
        if (key == -1) {
            // Generate new key 
            key = label_map.size();
            label_map[key] = label_str;
        }
        labels.push_back(key);

        // filename
        filenames.push_back(datadir + "/" + filename);
    }
    closedir(dirp);
}

std::string extract_signer_name(const std::string &dirpath) {
    
    std::string label = dirpath;

    // cut out sub-directory names in path
    size_t i = label.rfind('/', label.length());
    if (i != std::string::npos) {
        label = label.substr(i, label.length() - i);
    }

    // remove first all non-alphabet characters
    std::string temp = "";
    for (int j = 0; j < label.size(); j++) {
        if ((label[j] >= 'a' && label[j] <= 'z') ||
            (label[j] >= 'A' && label[j] <= 'Z')) {
            temp = temp + label[j];
        }
    }

    label = temp;

    return label;
}

std::string extract_sign_type(const std::string &filename) {
        std::string label = filename;
        size_t i = label.rfind('.', label.length());
        if (i != std::string::npos) {
            label = label.substr(0, i-1);
        }

        return label;
}

