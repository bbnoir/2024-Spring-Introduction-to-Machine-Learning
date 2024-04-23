#include "dis_model.hpp"

DisModel::DisModel(int n_features, int n_classes) {
    this->n_features = n_features;
    this->n_classes = n_classes;
    this->weights = new matrix_t(n_classes, n_features+1);
    this->dl_train = nullptr;
    this->dl_test = nullptr;
}

matrix_t DisModel::OneHot(vector_t* t) {
    int n_samples = t->size();
    matrix_t one_hot = matrix_t::Zero(n_samples, n_classes);
    for (int i = 0; i < n_samples; i++)
        one_hot(i, int((*t)(i))) = 1;
    return one_hot;
}

matrix_t DisModel::DesignMatrix(matrix_t* x) {
    matrix_t design_x = matrix_t::Ones(x->rows(), x->cols() + 1);
    design_x.block(0, 1, x->rows(), x->cols()) = *x;
    return design_x;
}

matrix_t DisModel::Softmax(matrix_t* x) {
    matrix_t exp_x = x->array().exp();
    matrix_t sum_exp_x = exp_x.rowwise().sum();
    matrix_t softmax_x = exp_x.array() / sum_exp_x.array().replicate(1, n_classes);
    return softmax_x;
}

vector_t DisModel::Predict(matrix_t* x) {
    vector_t pred = vector_t::Zero(x->rows());
    for (int i = 0; i < x->rows(); i++)
        x->row(i).maxCoeff(&pred(i));
    return pred;
}

void DisModel::Train(DataLoader* dl_train) {
    this->dl_train = dl_train;
    matrix_t x = *dl_train->x;
    vector_t t = *dl_train->t;
    int n_samples = dl_train->n_samples;
    matrix_t one_hot = OneHot(&t);
    matrix_t design_x = DesignMatrix(&x);
    matrix_t weights = matrix_t::Zero(n_classes, n_features + 1);
    // std::string filename = "results/dis_train_" + std::to_string(n_classes) + ".csv";
    // std::ofstream fout(filename);
    // fout << "Iteration,Accuracy\n";
    for (int i = 0; i < 10; i++) {
        matrix_t y = design_x * weights.transpose();
        matrix_t softmax_y = Softmax(&y);
        matrix_t error = softmax_y - one_hot;
        matrix_t grad = design_x.transpose() * error;
        for (int j = 0; j < n_classes; j++) {
            matrix_t diagnol_y = (softmax_y.col(j).array() * (1 - softmax_y.col(j).array())).matrix().asDiagonal();
            matrix_t hessian = design_x.transpose() * diagnol_y * design_x;
            weights.row(j) -= hessian.completeOrthogonalDecomposition().pseudoInverse() * grad.col(j);
        }
        // this->weights = &weights;
        // fout << i+1 << "," << (GenPredict(dl_train).array() == t.array()).count() / double(n_samples) << "\n";
    }
    this->weights = new matrix_t(weights);
    // dl_test = dl_train;
    // vector_t pred = GenPredict(dl_test);
    // double accuracy = (pred.array() == (*dl_test->t).array()).count() / double(dl_test->n_samples);
    // std::string filename2 = "results/dis_train_confusion_" + std::to_string(n_classes) + ".csv";
    // std::ofstream fout2(filename2);
    // for (int i = 0; i < n_classes; i++) {
    //     for (int j = 0; j < n_classes-1; j++)
    //         fout2 << ConfusionMatrix(dl_test->t, &pred)(i, j) << ",";
    //     fout2 << ConfusionMatrix(dl_test->t, &pred)(i, n_classes-1) << "\n";
    // }
}

vector_t DisModel::Test(DataLoader* dl_test) {
    vector_t pred = GenPredict(dl_test);
    double accuracy = (pred.array() == (*dl_test->t).array()).count() / double(dl_test->n_samples);
    std::cout << "Accuracy: " << accuracy << std::endl;
    std::cout << "Confusion Matrix:" << std::endl;
    std::cout << ConfusionMatrix(dl_test->t, &pred) << std::endl;
    // std::string filename = "results/dis_test_confusion_" + std::to_string(n_classes) + ".csv";
    // std::ofstream fout(filename);
    // for (int i = 0; i < n_classes; i++) {
    //     for (int j = 0; j < n_classes-1; j++)
    //         fout << ConfusionMatrix(dl_test->t, &pred)(i, j) << ",";
    //     fout << ConfusionMatrix(dl_test->t, &pred)(i, n_classes-1) << "\n";
    // }
    return pred;
}

vector_t DisModel::GenPredict(DataLoader* dl_test) {
    this->dl_test = dl_test;
    matrix_t x = *dl_test->x;
    vector_t t = *dl_test->t;
    int n_samples = dl_test->n_samples;
    matrix_t design_x = DesignMatrix(&x);
    matrix_t y = design_x * (*weights).transpose();
    matrix_t softmax_y = Softmax(&y);
    vector_t pred = Predict(&softmax_y);
    return pred;
}

matrix_t DisModel::ConfusionMatrix(vector_t* t, vector_t* y)
{
    matrix_t cm(n_classes, n_classes);
    cm.setZero();
    int n_samples = t->size();
    for (int i = 0; i < n_samples; i++)
        cm(int((*t)(i)), int((*y)(i)))++;
    return cm;
}
