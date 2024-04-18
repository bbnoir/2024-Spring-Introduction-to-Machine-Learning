#include "include.hpp"

class DataLoader {
public:
    DataLoader(const std::string& filename);

private:
    matrix_t* x;
    vector_t* t;
    int n_samples;
};