#include <iostream>
#include <fstream>
#include <Eigen/Dense>
 
typedef double scalar_t;
typedef Eigen::Matrix<scalar_t, Eigen::Dynamic, 1> vector_t;
typedef Eigen::Array<scalar_t, Eigen::Dynamic, Eigen::Dynamic> array_t;
typedef Eigen::Matrix<scalar_t, Eigen::Dynamic, Eigen::Dynamic> matrix_t;
 
int main()
{
  // Data
  matrix_t train_x(10000, 11);
  vector_t train_t(10000);
  matrix_t test_x(5817, 11);
  vector_t test_t(5817);

  // Input
  std::ifstream in_file("./HW1.csv");
  std::string line;
  std::getline(in_file, line); // skip the first line
  for (int i = 0; i < 10000; ++i)
  {
    std::getline(in_file, line);
    std::stringstream ss(line);
    std::string item;
    std::getline(ss, item, ',');
    train_t(i) = std::stod(item);
    for (int j = 0; j < 11; ++j)
    {
      std::getline(ss, item, ',');
      train_x(i, j) = std::stod(item);
    }
  }
  for (int i = 0; i < 5817; ++i)
  {
    std::getline(in_file, line);
    std::stringstream ss(line);
    std::string item;
    std::getline(ss, item, ',');
    test_t(i) = std::stod(item);
    for (int j = 0; j < 11; ++j)
    {
      std::getline(ss, item, ',');
      test_x(i, j) = std::stod(item);
    }
  }
  
  // Normalization
  vector_t mean = train_x.colwise().mean();
  vector_t std_dev = ((train_x.rowwise() - mean.transpose()).array().square().colwise().sum() / (train_x.rows()-1)).sqrt();
  train_x = (train_x.rowwise() - mean.transpose()).array().rowwise() / std_dev.transpose().array();
  mean = test_x.colwise().mean();
  std_dev = ((test_x.rowwise() - mean.transpose()).array().square().colwise().sum() / (test_x.rows()-1)).sqrt();
  test_x = (test_x.rowwise() - mean.transpose()).array().rowwise() / std_dev.transpose().array();


  // Training

  // Testing

  // Output
}
