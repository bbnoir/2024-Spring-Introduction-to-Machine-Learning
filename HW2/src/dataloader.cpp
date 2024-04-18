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
    for (int i = 0; i < n_samples; i++) {
        std::istringstream iss(lines[i]);
        std::string item;
        std::getline(iss, item, ',');
        (*t)(i) = std::stod(item);
        std::getline(iss, item, ',');
        (*x)(i, 0) = std::stod(item);
        std::getline(iss, item, ',');
        (*x)(i, 1) = std::stod(item);
    }
    std::cout << "DataLoader: loaded " << n_samples << " samples" << std::endl;
    // show last 5 samples
    std::cout << "DataLoader: last 5 samples" << std::endl;
    for (int i = n_samples - 5; i < n_samples; i++)
        std::cout << "t=" << (*t)(i) << ", x=(" << (*x)(i, 0) << ", " << (*x)(i, 1) << ")" << std::endl;
}