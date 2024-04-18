#pragma once
#include "include.hpp"

matrix_t ConfusionMatrix(vector_t* t, vector_t* y, int n_classes)
{
    matrix_t cm(n_classes, n_classes);
    cm.setZero();
    int n_samples = t->size();
    for (int i = 0; i < n_samples; i++)
        cm(int((*t)(i)), int((*y)(i)))++;
    return cm;
}