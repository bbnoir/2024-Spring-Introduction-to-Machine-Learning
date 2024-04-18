#include "gen_model.hpp"
#include "utils.hpp"

GenModel::GenModel(int n_features, int n_classes) {
    this->n_features = n_features;
    this->n_classes = n_classes;
    this->weights = new matrix_t(n_features, n_classes);
    this->bias = new vector_t(n_classes);
    this->dl_train = nullptr;
    this->dl_test = nullptr;
}

matrix_t GenModel::Mean(matrix_t* x, vector_t* t) {
    matrix_t mean = matrix_t::Zero(n_classes, n_features);
    int n_samples = dl_train->n_samples;
    std::vector<int> n_samples_per_class = dl_train->n_samples_per_class;
    for (int i = 0; i < n_samples; i++)
        mean.row((*t)(i)) += x->row(i);
    for (int i = 0; i < n_classes; i++)
        mean.row(i) /= n_samples_per_class[i];
    return mean;
}

matrix_t GenModel::Cov(matrix_t* x, vector_t* t) {
    matrix_t cov = matrix_t::Zero(n_features, n_features);
    int n_samples = dl_train->n_samples;
    matrix_t mean = x->colwise().mean();
    for (int i = 0; i < n_samples; i++)
        cov += (x->row(i) - mean).transpose() * (x->row(i) - mean);
    cov /= n_samples-1;
    return cov;
}

matrix_t GenModel::Softmax(matrix_t* x) {
    matrix_t exp_x = x->array().exp();
    matrix_t sum_exp_x = exp_x.rowwise().sum();
    matrix_t softmax_x = exp_x.array() / sum_exp_x.array().replicate(1, n_classes);
    return softmax_x;
}

vector_t GenModel::Predict(matrix_t* x) {
    vector_t pred = vector_t::Zero(x->rows());
    for (int i = 0; i < x->rows(); i++)
        x->row(i).maxCoeff(&pred(i));
    return pred;
}

void GenModel::Train(DataLoader* dl_train) {
    this->dl_train = dl_train;
    matrix_t mean = Mean(dl_train->x, dl_train->t);
    matrix_t cov = Cov(dl_train->x, dl_train->t);
    weights = new matrix_t(cov.inverse() * mean.transpose());
    vector_t priors = (*dl_train->priors).array().log();
    bias = new vector_t(n_classes);
    for (int i = 0; i < n_classes; i++)
        (*bias)(i) = -0.5 * mean.row(i) * cov.inverse() * mean.row(i).transpose() + priors(i);
}

void GenModel::Test(DataLoader* dl_test) {
    this->dl_test = dl_test;
    matrix_t x = *dl_test->x;
    vector_t t = *dl_test->t;
    int n_samples = dl_test->n_samples;
    matrix_t y = x * (*weights) + (*bias).transpose().replicate(n_samples, 1);
    y = Softmax(&y);
    vector_t pred = Predict(&y);
    double accuracy = (pred.array() == t.array()).count() / double(n_samples);
    std::cout << "Accuracy: " << accuracy << std::endl;
    std::cout << "Confusion Matrix:" << std::endl;
    std::cout << ConfusionMatrix(&t, &pred, n_classes) << std::endl;
}