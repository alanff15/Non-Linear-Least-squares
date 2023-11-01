#pragma once

#include <Eigen/Eigen>

#define ERROR_TYPE double
#define VECTOR_TYPE Eigen::Matrix<ERROR_TYPE, -1, 1>
#define MATRIX_TYPE Eigen::Matrix<ERROR_TYPE, -1, -1>

#define DEFAULT_MAX_ITR 500
#define DEFAULT_EPSILON (ERROR_TYPE)(1e-8)
#define DEFAULT_DELTA (ERROR_TYPE)(1e-5)

class NonLinearLSQ {
public:
  NonLinearLSQ();
  ~NonLinearLSQ();

  void setErrorFunction(const std::function<ERROR_TYPE(const VECTOR_TYPE&, const std::vector<ERROR_TYPE>&)>& func);
  void setData(const std::vector<std::vector<ERROR_TYPE>>& data_in);
  void solve(const VECTOR_TYPE& param_gess);
  VECTOR_TYPE getParam();

private:
  ERROR_TYPE derror_dx(int dx_index, const VECTOR_TYPE& param, const std::vector<ERROR_TYPE>& data_line);
  void getSystem(const VECTOR_TYPE& param, MATRIX_TYPE& H, VECTOR_TYPE& b);

  VECTOR_TYPE delta;
  VECTOR_TYPE final_param;
  const std::vector<std::vector<ERROR_TYPE>>* data;
  std::function<ERROR_TYPE(const VECTOR_TYPE&, const std::vector<ERROR_TYPE>&)> error;
};