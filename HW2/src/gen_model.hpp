#pragma once
#include "include.hpp"
#include "dataloader.hpp"

class GenModel {
public:
    GenModel(int n_features, int n_classes);
    void Train(DataLoader* dl_train);
    vector_t Test(DataLoader* dl_test);

    matrix_t Mean(matrix_t* x, vector_t* t);
    matrix_t Cov(matrix_t* x, vector_t* t);
    matrix_t Softmax(matrix_t* x);
    vector_t Predict(matrix_t* x);

public:
    int n_features;
    int n_classes;
    matrix_t* weights;
    vector_t* bias;
    DataLoader* dl_train;
    DataLoader* dl_test;
};