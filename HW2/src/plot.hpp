#pragma once
#include "include.hpp"
#include "dataloader.hpp"

DataLoader* GenPlotData(const int resolution) {
    matrix_t* x = new matrix_t(resolution * resolution, 2);
    vector_t* t = new vector_t(resolution * resolution);
    t->setZero();
    scalar_t scale = 100.0 / resolution;
    for (int i = 0; i < resolution; i++) {
        for (int j = 0; j < resolution; j++) {
            (*x)(i * resolution + j, 0) = i * scale;
            (*x)(i * resolution + j, 1) = j * scale;
        }
    }
    return new DataLoader(x, t);
}

void WritePlotData(vector_t* y, DataLoader* dl, const std::string& filename) {
    std::ofstream fout(filename);
    fout << "y" << '\n';
    for (int i = 0; i < dl->n_samples; i++)
        fout << (*y)(i) << '\n';
}