#include <iostream>
#include <fstream>
#include <cmath>
#include <Eigen/Dense>
#include <Eigen/QR>
 
typedef double scalar_t;
typedef Eigen::Matrix<scalar_t, Eigen::Dynamic, 1> vector_t;
typedef Eigen::Array<scalar_t, Eigen::Dynamic, Eigen::Dynamic> array_t;
typedef Eigen::Matrix<scalar_t, Eigen::Dynamic, Eigen::Dynamic> matrix_t;

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
  for (int i = 0; i < 15818; ++i)
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
  
  // Normalization
  vector_t mean = train_x.colwise().mean();
  vector_t std_dev = ((train_x.rowwise() - mean.transpose()).array().square().colwise().sum() / (train_x.rows()-1)).sqrt();
  train_x = (train_x.rowwise() - mean.transpose()).array().rowwise() / std_dev.transpose().array();
  test_x = (test_x.rowwise() - mean.transpose()).array().rowwise() / std_dev.transpose().array();

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
  {
    matrix_t phi = design_matrix(test_x, M_list[i], s, u.row(i));
    test_y[i] = phi * w[i];
    test_mse[i] = cal_mse(test_t, test_y[i]);
    test_accuracy[i] = cal_accuracy(test_t, test_y[i]);
  }

  // Output
  std::ofstream out_file("./output.csv");
  out_file << "M,Train MSE,Train Accuracy,Test MSE,Test Accuracy\n";
  for (int i = 0; i < 6; ++i)
    out_file << M_list[i] << "," << train_mse[i] << "," << train_accuracy[i] << "," << test_mse[i] << "," << test_accuracy[i] << "\n";
  out_file.close();
  out_file.open("./pred.csv");
  out_file << "M=5,M=10,M=15,M=20,M=25,M=30\n";
  for (int i = 0; i < train_y[0].size(); ++i)
  {
    for (int j = 0; j < 6; ++j)
      out_file << train_y[j](i) << ",";
    out_file << "\n";
  }
  for (int i = 0; i < test_y[0].size(); ++i)
  {
    for (int j = 0; j < 6; ++j)
      out_file << test_y[j](i) << ",";
    out_file << "\n";
  }
}
