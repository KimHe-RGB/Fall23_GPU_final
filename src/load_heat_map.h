/**
 * @file load_heat_map.h
 * 
 * @author Xujin He (xh1131@nyu.edu)
 * @brief This is sequential io code to load/write heat map into/from csv file
 * @version 0.1
 * @date 2023-12-06
 * 
 * @copyright Copyright (c) 2023
 * 
 */
#include <iostream>
#include <fstream>
#include <sstream>
#include <limits>
#include <iomanip>

/**
 * @brief 
 * 
 * @param filename 
 * @param heat m x n heat map
 * @param dim size of heat map, m*n
 */
void loadCSV(const std::string& filename, double* heat, int dim) {

    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Unable to open file " << filename << std::endl;
        return;
    }

    std::string line;
    int ind = 0;
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        std::string value;

        while (std::getline(iss, value, ',')) {
            try {
                // Convert the string to a double and push it to the row
                heat[ind] = std::stod(value);
                ind++;
            } catch (const std::invalid_argument& e) {
                std::cerr << "Error: Invalid argument in CSV file." << std::endl;
                file.close();
                return;
            } catch (const std::out_of_range& e) {
                std::cerr << "Error: Out of range in CSV file." << std::endl;
                file.close();
                return;
            }
        }
    }
    file.close();
    std::cout << "CSV file " << filename << " successfully loaded." << std::endl;
    return;
}

/**
 * @brief write the heat map to a new csv
 * 
 */
void writeCSV(const std::string& filename, double* matrix, int width, int height) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Unable to open file " << filename << " for writing." << std::endl;
        return;
    }
    // Set the precision of the output stream to maximum float precision
    file << std::setprecision(std::numeric_limits<double>::max_digits10);

    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            file << matrix[i * width + j];
            // Add a comma for all elements except the last one in the row
            if (j < width - 1) {
                file << ",";
            }
        }
        file << std::endl; // Move to the next line after each row
    }

    file.close();
    std::cout << "CSV file " << filename << " successfully written." << std::endl;
}

/**
 * @brief render the heat map as a img
 * 
 */
void snapshot(const std::string& filename, double* matrix, int width, int height) {
    
}