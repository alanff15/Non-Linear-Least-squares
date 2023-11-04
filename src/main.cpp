#include <iostream>
#include "NonLinearLSQ.h"

// data: table of (y,x)
std::vector<std::vector<double>> data = {
  {0.2203, 0.000},  {0.0677, 0.250}, {-0.3814, 0.500}, {-0.2378, 0.750}, {0.1997, 1.000},  {0.3236, 1.250},  {-0.4040, 1.500}, {-0.0491, 1.750},
  {-0.1186, 2.000}, {0.1361, 2.250}, {-0.1237, 2.500}, {0.7372, 2.750},  {0.3649, 3.000},  {0.6857, 3.250},  {1.4986, 3.500},  {1.7044, 3.750},
  {1.5726, 4.000},  {2.0807, 4.250}, {2.5034, 4.500},  {2.9446, 4.750},  {2.5311, 5.000},  {2.1367, 5.250},  {2.1622, 5.500},  {1.7561, 5.750},
  {1.0682, 6.000},  {0.6799, 6.250}, {1.0680, 6.500},  {0.4497, 6.750},  {-0.0410, 7.000}, {-0.0995, 7.250}, {-0.1420, 7.500}, {-0.1194, 7.750},
  {-0.2733, 8.000}, {0.4678, 8.250}, {-0.3094, 8.500}, {-0.4456, 8.750}, {0.1172, 9.000},  {-0.0850, 9.250}, {0.0185, 9.500},  {-0.0313, 9.750},
};

// error func.: y = f(p,x) | err = (y-f(p,x))²
double err(const double p[], const std::vector<double> yx) {
  double ret = yx[0];                                 // y
  ret -= p[0] * exp(-pow((yx[1] - p[1]) * p[2], 2));  // f(p,x)
  return ret * ret;                                   // err
}

// kernel to reject outliers
double kernel(double error) {
  return exp(-error * error * 10);
}

int main() {
  NonLinearLSQ<double, 3> lsq;  // create solver <data type, number of params to find>
  lsq.setData(data);            // define data
  lsq.setErrorFunction(err);    // define error function

  // optional configs (converge faster / avoid divergency)
  lsq.setKernel(kernel);     // define kernel
  lsq.setLimits(0, 1, 4);    // limit param 0
  lsq.setLimits(1, 3, 6);    // limit param 1
  lsq.setLimits(2, 0.5, 2);  // limit param 2

  // curve fit
  lsq.solve();

  // show result
  std::cout << "Params: " << lsq.getParam().transpose() << std::endl;
  std::cout << "Iter.:  " << lsq.getLastIterations() << std::endl;
  std::cout << "Error:  " << lsq.getError() << std::endl;

  return EXIT_SUCCESS;
}