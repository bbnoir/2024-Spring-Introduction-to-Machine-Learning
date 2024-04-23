#pragma once
#include "include.hpp"
#include "dataloader.hpp"

class DisModel {
public:
    DisModel(int n_features, int n_classes);
    void Train(DataLoader* dl_train);
    vector_t Test(DataLoader* dl_test);
    vector_t GenPredict(DataLoader* dl_test);

    matrix_t OneHot(vector_t* t);
    matrix_t DesignMatrix(matrix_t* x);
    matrix_t Softmax(matrix_t* x);
    matrix_t ConfusionMatrix(vector_t* t, vector_t* y);
    vector_t Predict(matrix_t* x);

public:
    int n_features;
    int n_classes;
    matrix_t* weights;
    DataLoader* dl_train;
    DataLoader* dl_test;
};