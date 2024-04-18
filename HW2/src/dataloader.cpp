#include "dataloader.hpp"

DataLoader::DataLoader(const std::string& filename) {
    std::ifstream fin(filename);
    std::string line;
    std::getline(fin, line); // skip header
    std::vector<std::string> lines;
    while (std::getline(fin, line))
        lines.emplace_back(line);

    n_samples = lines.size();
    x = new matrix_t(n_samples, 2);
    t = new vector_t(n_samples);
    n_samples_per_class.clear();
    for (int i = 0; i < 4; i++)
        n_samples_per_class.emplace_back(0);
    for (int i = 0; i < n_samples; i++) {
        std::istringstream iss(lines[i]);
        std::string item;
        std::getline(iss, item, ',');
        (*t)(i) = std::stod(item);
        std::getline(iss, item, ',');
        (*x)(i, 0) = std::stod(item);
        std::getline(iss, item, ',');
        (*x)(i, 1) = std::stod(item);
        n_samples_per_class[int((*t)(i))]++;
    }
    priors = new vector_t(4);
    for (int i = 0; i < 4; i++)
        (*priors)(i) = n_samples_per_class[i] / double(n_samples);
}

DataLoader::DataLoader(matrix_t* x, vector_t* t) {
    this->x = x;
    this->t = t;
    n_samples = x->rows();
    n_samples_per_class.clear();
    for (int i = 0; i < 4; i++)
        n_samples_per_class.emplace_back(0);
    priors = nullptr;
}
