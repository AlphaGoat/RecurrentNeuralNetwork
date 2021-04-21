#include <dirent.h>
#include <string>
#include <vector>
#include <map>
#include "utils.h"

#include <iostream>

void get_vowel_data(const std::string &datadir,
        std::vector<std::string> &filenames,
        std::vector<int> &labels,
        std::map<int, std::string> &label_map) {

    // Get data files labeled by the identity of the signed
    DIR* dirp = opendir(datadir.c_str());
    struct dirent *dp;
    while ((dp = readdir(dirp)) != NULL) {

        /* Check that the filename isn't just 'this' directory 
         * or 'prev' */
        std::string filename = dp->d_name;
        if (filename == "." || filename == "..") 
            continue;

        else if (filename == "ae.train" || filename == "ae.test") 
            continue;

//        std::string label_str;
//        if (label_type_flag == SIGN_ID) {
//            label_str = extract_sign_type(filename);
//        }
//
//        else {
//            label_str = extract_signer_name(datadir); 
//        }
        
        /* Get label by cutting out all directory information */
        size_t i = datadir.rfind('/', datadir.length());
        std::string label;
        if (i != std::string::npos) {
            label = datadir.substr(i, datadir.length() - i); 
            label = label.substr(1, datadir.length());
        }

        if (label == "." || label == "..") {
            continue;
        }

        // Add label to hashmap, if it is not already there.
        // Otherwise, assign an integer label to data if it is
        int key = find_by_value(label_map, label);
        if (key == -1) {
            // Generate new key 
            key = label_map.size();
            label_map[key] = label;
        }
        labels.push_back(key);

        // filename
        filenames.push_back(datadir + "/" + filename);
    }
    closedir(dirp);
}
