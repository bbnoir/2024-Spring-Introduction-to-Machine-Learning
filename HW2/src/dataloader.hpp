#pragma once
#include "include.hpp"

class DataLoader {
public:
    DataLoader(const std::string& filename);
    DataLoader(matrix_t* x, vector_t* t);

public:
    matrix_t* x;
    vector_t* t;
    int n_samples;
    std::vector<int> n_samples_per_class;
    vector_t* priors;
};