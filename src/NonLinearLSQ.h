#pragma once

#include <Eigen/Eigen>

#define PARAM_TYPE Eigen::Vector2d
#define MATRIX_TYPE Eigen::MatrixXd
#define ERROR_TYPE double

#define DEFAULT_MAX_ITR 500
#define DEFAULT_EPSILON (ERROR_TYPE)(1e-8)
#define DEFAULT_DELTA (ERROR_TYPE)(1e-5)

class NonLinearLSQ {
public:
  NonLinearLSQ();
  ~NonLinearLSQ();

  void setErrorFunction(const std::function<ERROR_TYPE(const PARAM_TYPE&, const std::vector<ERROR_TYPE>&)>& func);
  void setData(const std::vector<std::vector<ERROR_TYPE>>& data_in);
  void solve(const PARAM_TYPE& param_gess);
  PARAM_TYPE getParam();

private:
  ERROR_TYPE derror_dx(int dx_index, const PARAM_TYPE& param, const std::vector<ERROR_TYPE>& data_line);
  void getSystem(const PARAM_TYPE& param, MATRIX_TYPE& H, PARAM_TYPE& b);

  PARAM_TYPE delta;
  PARAM_TYPE final_param;
  const std::vector<std::vector<ERROR_TYPE>>* data;
  std::function<ERROR_TYPE(const PARAM_TYPE&, const std::vector<ERROR_TYPE>&)> error;
};