/* Implements median filter for Auslan dataset to remove 
 * x,y,z values that dropped out 
 */
#include <iostream>
#include "sign_utils.h"
#include "utils.h"

sign_data_instance_t median_filter_1D(sign_data_instance_t &data_instance) {

    /* go through each column and, if 0, replace with median 
     * of neighboring values (number of neighbors used depends
     * on window size */

    /* initialize window */
    std::array <float, MEDIAN_WINDOW_SIZE> window; // see utils.h for MEDIAN_WINDOW_SIZE
    const unsigned int NUM_COLS = data_instance[0].size();
    const unsigned int NUM_ROWS = data_instance.size();

    /* Copy data instance into new data structure */
    sign_data_instance_t new_instance = data_instance;

    for (int i = 0; i < NUM_COLS; i++) {

        for (int j = 0; j < NUM_ROWS; j++) {

            // If we can't fill while filter with values in dataset, pad with 0's
            if (j < MEDIAN_WINDOW_SIZE / 2) {
                
                int num_pad = MEDIAN_WINDOW_SIZE / 2 - j;
                for (int k = 0; k < num_pad; k++) {

                    window[k] = 0.0;
                }

                for (int h = 0; h < MEDIAN_WINDOW_SIZE - num_pad; h++) {

                    window[h] = data_instance[h][i];
                }
            }

            else if (NUM_ROWS - (j + 1) < MEDIAN_WINDOW_SIZE / 2) {
                
                int num_pad = (MEDIAN_WINDOW_SIZE / 2) - (NUM_ROWS - j);
                for (int k = 0; k < num_pad; k++) {

                    window[MEDIAN_WINDOW_SIZE - k] = 0.0;
                }

                int start_index = j - MEDIAN_WINDOW_SIZE / 2;
                for (int h = 0; h < MEDIAN_WINDOW_SIZE - num_pad;  h++) {

                    window[h] = data_instance[start_index + h][i]; 
                }
            }

            else {

                int vector_index = 0;
                for (int k = j - MEDIAN_WINDOW_SIZE / 2;
                        k <= j + MEDIAN_WINDOW_SIZE / 2;
                        k++) {

                    window[vector_index] = data_instance[k][i];
                    vector_index++;
                }
            }

            /* order values in filter */
            for (int m = 0; m < MEDIAN_WINDOW_SIZE; m++) {
                for (int n = m + 1; n < MEDIAN_WINDOW_SIZE; n++) {
                    if (window[m] > window[n]) {
                        float temp = window[m];
                        window[m] = window[n];
                        window[n] = temp;
                    }
                }
            } 

            /* Select median value */
            float median_val;
            if (MEDIAN_WINDOW_SIZE % 2 == 0) {
               median_val =  (window[MEDIAN_WINDOW_SIZE / 2] +
                   window[(MEDIAN_WINDOW_SIZE / 2) - 1]) / 2;
            }

            else {
                median_val = window[MEDIAN_WINDOW_SIZE / 2];
            }

            /* Finally, insert the value for j in the copied array */
            new_instance[j][i] = median_val;
        }
    }

    return new_instance;
}
