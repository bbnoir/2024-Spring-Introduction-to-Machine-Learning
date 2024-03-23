#include <iostream>
#include <fstream>
#include <cmath>
#include <filesystem>
#include <vector>
#include <Eigen/Dense>
 
typedef double scalar_t;
typedef Eigen::Matrix<scalar_t, Eigen::Dynamic, 1> vector_t;
typedef Eigen::Array<scalar_t, Eigen::Dynamic, Eigen::Dynamic> array_t;
typedef Eigen::Matrix<scalar_t, Eigen::Dynamic, Eigen::Dynamic> matrix_t;

std::string out_dir = "./output/new_test/";

// Parameters
int N = 11; // number of features
const int M_list[6] = {5, 10, 15, 20, 25, 30};
const scalar_t s = 0.1;
matrix_t u(6, 30);

void init_parameters()
{
  for (int i = 0; i < 6; ++i)
    for (int j = 1; j < M_list[i]; ++j)
    {
      scalar_t d_j = j, d_M = M_list[i];
      u(i, j) = 3 * (-d_M + 1 + 2*(d_j-1)*(d_M-1)/(d_M-2)) / d_M;
    }
}

scalar_t cal_mse(const vector_t &t, const vector_t &y)
{
  return (t - y).array().square().sum() / t.size();
}

scalar_t cal_accuracy(const vector_t &t, const vector_t &y)
{
  scalar_t accuracy = 0;
  for (int i = 0; i < t.size(); ++i)
  {
    if (t(i) == 0)
      accuracy += std::abs(y(i) - 1);
    else
      accuracy += std::abs((y(i) - t(i)) / t(i));
  }
  accuracy /= t.size();
  return 1 - accuracy;
}

matrix_t design_matrix(const matrix_t &x, const int M, const scalar_t s, const vector_t &u)
{
  int N = x.cols();
  matrix_t phi(x.rows(), N*M);
  for (int i = 0; i < N; ++i)
  {
    phi.col(i*M) = vector_t::Ones(x.rows());
    for (int j = 1; j < M; ++j)
      phi.col(i*M+j) = 1 / (1 + (-(x.col(i).array() - u(j)) / s).exp());
  }
  return phi;
}

// test
void test(const vector_t &w, const matrix_t &x, const vector_t &t, vector_t &y, scalar_t &mse, scalar_t &accuracy, const int M_idx)
{
  matrix_t phi = design_matrix(x, M_list[M_idx], s, u.row(M_idx));
  y = phi * w;
  mse = cal_mse(t, y);
  accuracy = cal_accuracy(t, y);
}

void show_accuracy(const scalar_t *train_mse, const scalar_t *train_accuracy, const scalar_t *test_mse, const scalar_t *test_accuracy, const scalar_t *demo_mse, const scalar_t *demo_accuracy)
{
  std::cout << "| M\t| Train MSE\t| Train Accuracy\t| Test MSE\t| Test Accuracy\t\t| Demo MSE\t| Demo Accuracy\n";
  std::cout << "----------------------------------------------------------------------------------------------------------------------\n";
  for (int i = 0; i < 6; ++i)
    std::cout << "| " << M_list[i] << "\t| " << train_mse[i] << "\t| " << train_accuracy[i] << "\t\t| " << test_mse[i] << "\t| " << test_accuracy[i] << "\t\t| " << demo_mse[i] << "\t| " << demo_accuracy[i] << "\n";
  std::cout << "----------------------------------------------------------------------------------------------------------------------\n";
}

void export_accuracy(std::string filename, const scalar_t *train_mse, const scalar_t *train_accuracy, const scalar_t *test_mse, const scalar_t *test_accuracy)
{
  std::ofstream out_file(filename);
  out_file << "M,Train MSE,Train Accuracy,Test MSE,Test Accuracy\n";
  for (int i = 0; i < 6; ++i)
    out_file << M_list[i] << "," << train_mse[i] << "," << train_accuracy[i] << "," << test_mse[i] << "," << test_accuracy[i] << "\n";
  out_file.close();
}

void export_prediction(std::string filename, const vector_t *pred)
{
  std::ofstream out_file(filename);
  out_file << "M=5,M=10,M=15,M=20,M=25,M=30\n";
  for (int i = 0; i < pred[0].size(); ++i)
  {
    for (int j = 0; j < 6; ++j)
      out_file << pred[j](i) << ",";
    out_file << "\n";
  }
  out_file.close();
}
 
int main()
{
  // Initialize Parameters
  init_parameters();

  // Data
  matrix_t train_x(10000, N);
  vector_t train_t(10000);
  matrix_t test_x(5817, N);
  vector_t test_t(5817);

  // Input
  std::ifstream in_file("./HW1.csv");
  std::string line;
  std::getline(in_file, line); // skip the first line
  for (int i = 0; i < 15817; ++i)
  {
    std::getline(in_file, line);
    std::stringstream ss(line);
    std::string item;
    std::getline(ss, item, ',');
    if (i < 10000)
      train_t(i) = std::stod(item);
    else
      test_t(i-10000) = std::stod(item);
    for (int j = 0; j < N; ++j)
    {
      std::getline(ss, item, ',');
      if (i < 10000)
        train_x(i, j) = std::stod(item);
      else
        test_x(i-10000, j) = std::stod(item);
    }
  }
  in_file.close();

  // Demo Data
  std::vector<std::string> demo_lines;
  in_file.open("./HW1_demo.csv");
  std::getline(in_file, line); // skip the first line
  while (std::getline(in_file, line))
    demo_lines.push_back(line);
  in_file.close();

  int demo_size = demo_lines.size();
  matrix_t demo_x(demo_size, N);
  vector_t demo_t(demo_size);
  for (int i = 0; i < demo_size; ++i)
  {
    std::stringstream ss(demo_lines[i]);
    std::string item;
    std::getline(ss, item, ',');
    demo_t(i) = std::stod(item);
    for (int j = 0; j < N; ++j)
    {
      std::getline(ss, item, ',');
      demo_x(i, j) = std::stod(item);
    }
  }
  
  // Normalization
  vector_t mean = train_x.colwise().mean();
  vector_t std_dev = ((train_x.rowwise() - mean.transpose()).array().square().colwise().sum() / (train_x.rows()-1)).sqrt();
  train_x = (train_x.rowwise() - mean.transpose()).array().rowwise() / std_dev.transpose().array();
  test_x = (test_x.rowwise() - mean.transpose()).array().rowwise() / std_dev.transpose().array();
  demo_x = (demo_x.rowwise() - mean.transpose()).array().rowwise() / std_dev.transpose().array();

  // Part: Normal Linear Regression

  // Training
  vector_t w[6];
  vector_t train_y[6];
  scalar_t train_mse[6];
  scalar_t train_accuracy[6];
  for (int i = 0; i < 6; ++i)
  {
    matrix_t phi = design_matrix(train_x, M_list[i], s, u.row(i));
    w[i] = (phi.transpose() * phi).ldlt().solve(phi.transpose() * train_t);
    // w[i] = phi.completeOrthogonalDecomposition().pseudoInverse() * train_t;
    train_y[i] = phi * w[i];
    train_mse[i] = cal_mse(train_t, train_y[i]);
    train_accuracy[i] = cal_accuracy(train_t, train_y[i]);
  }

  // Testing
  vector_t test_y[6];
  scalar_t test_mse[6];
  scalar_t test_accuracy[6];
  for (int i = 0; i < 6; ++i)
    test(w[i], test_x, test_t, test_y[i], test_mse[i], test_accuracy[i], i);

  // Demo
  vector_t demo_y[6];
  scalar_t demo_mse[6];
  scalar_t demo_accuracy[6];
  for (int i = 0; i < 6; ++i)
    test(w[i], demo_x, demo_t, demo_y[i], demo_mse[i], demo_accuracy[i], i);

  // Show Accuracy
  show_accuracy(train_mse, train_accuracy, test_mse, test_accuracy, demo_mse, demo_accuracy);

  // Output
  std::filesystem::create_directories(out_dir);
  export_accuracy(out_dir + "accuracy.csv", train_mse, train_accuracy, test_mse, test_accuracy);
  export_prediction(out_dir + "train_prediction.csv", train_y);
  export_prediction(out_dir + "test_prediction.csv", test_y);

  // Part: 5 Fold Cross Validation

  // Split Data into 5 Folds
  matrix_t split_train_x[5];
  vector_t split_train_t[5];
  for (int i = 0; i < 5; ++i)
  {
    split_train_x[i] = train_x.block(i*2000, 0, 2000, N);
    split_train_t[i] = train_t.segment(i*2000, 2000);
  }

  // Training
  vector_t w_cv[5][6];
  vector_t train_y_cv[5][6];
  scalar_t train_mse_cv[5][6];
  scalar_t train_accuracy_cv[5][6];
  vector_t val_y_cv[5][6];
  scalar_t val_mse_cv[5][6];
  scalar_t val_accuracy_cv[5][6];
  scalar_t accuracy_sum[6] = {0};
  int best_M_idx = 0;
  for (int i = 0; i < 6; ++i)
  {
    for (int j = 0; j < 5; ++j)
    {
      matrix_t merge_train_x(8000, N);
      vector_t merge_train_t(8000);
      int k = 0;
      for (int l = 0; l < 5; ++l)
      {
        if (l != j)
        {
          merge_train_x.block(k*2000, 0, 2000, N) = split_train_x[l];
          merge_train_t.segment(k*2000, 2000) = split_train_t[l];
          k++;
        }
      }
      matrix_t phi = design_matrix(merge_train_x, M_list[i], s, u.row(i));
      w_cv[j][i] = (phi.transpose() * phi).ldlt().solve(phi.transpose() * merge_train_t);
      // w_cv[j][i] = phi.completeOrthogonalDecomposition().pseudoInverse() * merge_train_t;
      train_y_cv[j][i] = phi * w_cv[j][i];
      train_mse_cv[j][i] = cal_mse(merge_train_t, train_y_cv[j][i]);
      train_accuracy_cv[j][i] = cal_accuracy(merge_train_t, train_y_cv[j][i]);
      test(w_cv[j][i], split_train_x[j], split_train_t[j], val_y_cv[j][i], val_mse_cv[j][i], val_accuracy_cv[j][i], i);
      accuracy_sum[i] += val_accuracy_cv[j][i];
    }
    if (accuracy_sum[i] > accuracy_sum[best_M_idx])
      best_M_idx = i;
  }
  std::cout << "Best M: " << M_list[best_M_idx] << "\n";

  // Part: Ridge Regression

  // Training
  const int lambda_list_size = 1;
  scalar_t lambda_list[6] = {0.1, 0.005, 0.001, 0.5, 1, 10};
  vector_t w_ridge[lambda_list_size][6];
  vector_t train_y_ridge[lambda_list_size][6];
  scalar_t train_mse_ridge[lambda_list_size][6];
  scalar_t train_accuracy_ridge[lambda_list_size][6];
  for (int i = 0; i < 6; ++i)
  {
    matrix_t phi = design_matrix(train_x, M_list[i], s, u.row(i));
    for (int j = 0; j < lambda_list_size; ++j)
    {
      w_ridge[j][i] = (phi.transpose() * phi + lambda_list[j] * matrix_t::Identity(M_list[i]*N, M_list[i]*N)).completeOrthogonalDecomposition().pseudoInverse() * phi.transpose() * train_t;
      train_y_ridge[j][i] = phi * w_ridge[j][i];
      train_mse_ridge[j][i] = cal_mse(train_t, train_y_ridge[j][i]);
      train_accuracy_ridge[j][i] = cal_accuracy(train_t, train_y_ridge[j][i]);
    }
  }

  // Testing
  vector_t test_y_ridge[lambda_list_size][6];
  scalar_t test_mse_ridge[lambda_list_size][6];
  scalar_t test_accuracy_ridge[lambda_list_size][6];
  for (int i = 0; i < 6; ++i)
    for (int j = 0; j < lambda_list_size; ++j)
      test(w_ridge[j][i], test_x, test_t, test_y_ridge[j][i], test_mse_ridge[j][i], test_accuracy_ridge[j][i], i);

  // Demo
  vector_t demo_y_ridge[lambda_list_size][6];
  scalar_t demo_mse_ridge[lambda_list_size][6];
  scalar_t demo_accuracy_ridge[lambda_list_size][6];
  for (int i = 0; i < 6; ++i)
    for (int j = 0; j < lambda_list_size; ++j)
      test(w_ridge[j][i], demo_x, demo_t, demo_y_ridge[j][i], demo_mse_ridge[j][i], demo_accuracy_ridge[j][i], i);

  // Show accuracy
  std::cout << "Ridge Regression\n";
  for (int i = 0; i < lambda_list_size; ++i)
  {
    std::cout << "Lambda: " << lambda_list[i] << "\n";
    show_accuracy(train_mse_ridge[i], train_accuracy_ridge[i], test_mse_ridge[i], test_accuracy_ridge[i], demo_mse_ridge[i], demo_accuracy_ridge[i]);
  }
  
}
