#pragma once

#include <Eigen/Eigen>

#define DEFAULT_MAX_ITR 500
#define DEFAULT_EPSILON (1e-8)
#define DEFAULT_DELTA (1e-5)

#define VEC_TYPE Eigen::Matrix<NUM_TYPE, N, 1>
#define MAT_TYPE Eigen::Matrix<NUM_TYPE, N, N>

template <typename NUM_TYPE, uint32_t N>
class NonLinearLSQ {
public:
  NonLinearLSQ();
  ~NonLinearLSQ();

  void setErrorFunction(const std::function<NUM_TYPE(const VEC_TYPE&, const std::vector<NUM_TYPE>&)>&);
  void setData(const std::vector<std::vector<NUM_TYPE>>&);
  void solve(const VEC_TYPE&);
  VEC_TYPE getParam();

  void setMaxIteration(int);
  void setEpsilon(NUM_TYPE);
  void setDelta(VEC_TYPE);
  int getMaxIteration();
  NUM_TYPE getEpsilon();
  VEC_TYPE getDelta();

private:
  NUM_TYPE derror_dx(int, const VEC_TYPE&, const std::vector<NUM_TYPE>&);
  void getSystem(const VEC_TYPE&, MAT_TYPE&, VEC_TYPE&);

  int max_iterations;
  NUM_TYPE epsilon;
  VEC_TYPE delta;
  VEC_TYPE final_param;
  const std::vector<std::vector<NUM_TYPE>>* data;
  std::function<NUM_TYPE(const VEC_TYPE&, const std::vector<NUM_TYPE>&)> error;
};

///////////////////////////////////////////////////////////////////////////////
// Implementation:

template <typename NUM_TYPE, uint32_t N>
NonLinearLSQ<NUM_TYPE, N>::NonLinearLSQ() {
  max_iterations = DEFAULT_MAX_ITR;
  epsilon = (NUM_TYPE)DEFAULT_EPSILON;
  for (int i = 0; i < delta.rows(); i++) delta(i) = (NUM_TYPE)DEFAULT_DELTA;
}

template <typename NUM_TYPE, uint32_t N>
NonLinearLSQ<NUM_TYPE, N>::~NonLinearLSQ() {}

template <typename NUM_TYPE, uint32_t N>
NUM_TYPE NonLinearLSQ<NUM_TYPE, N>::derror_dx(int dx_index, const VEC_TYPE& input, const std::vector<NUM_TYPE>& data_line) {
  VEC_TYPE input_delta = input;
  input_delta[dx_index] += delta[dx_index];
  return (error(input_delta, data_line) - error(input, data_line)) / delta[dx_index];
}

template <typename NUM_TYPE, uint32_t N>
void NonLinearLSQ<NUM_TYPE, N>::getSystem(const VEC_TYPE& input, MAT_TYPE& H, VEC_TYPE& b) {
  // zerar sistema
  for (int j, i = 0; i < H.rows(); i++) {
    for (j = 0; j < H.cols(); j++) {
      H(i, j) = 0;
    }
    b(i) = 0;
  }
  // varer dados
  MAT_TYPE Hs;
  VEC_TYPE bs;
  VEC_TYPE Jt;  // rows:IN_TYPE cols:OUT_TYPE
  NUM_TYPE err;
  for (int j, i = 0; i < data->size(); i++) {
    err = error(input, (*data)[i]);
    for (j = 0; j < Jt.rows(); j++) {
      Jt(j) = derror_dx(j, input, (*data)[i]);
    }
    Hs << Jt * Jt.transpose();
    bs << Jt * err;
    H += Hs;
    b += bs;
  }
}

template <typename NUM_TYPE, uint32_t N>
void NonLinearLSQ<NUM_TYPE, N>::setErrorFunction(const std::function<NUM_TYPE(const VEC_TYPE&, const std::vector<NUM_TYPE>&)>& func) {
  error = func;
}

template <typename NUM_TYPE, uint32_t N>
void NonLinearLSQ<NUM_TYPE, N>::setData(const std::vector<std::vector<NUM_TYPE>>& data_in) {
  data = &data_in;
}

template <typename NUM_TYPE, uint32_t N>
void NonLinearLSQ<NUM_TYPE, N>::solve(const VEC_TYPE& input_gess) {
  VEC_TYPE param = input_gess;
  MAT_TYPE H;
  VEC_TYPE b;
  VEC_TYPE delta_param;
  int itr = 1;
  do {
    getSystem(param, H, b);
    delta_param = H.colPivHouseholderQr().solve(-b);
    param += delta_param;
  } while ((delta_param.norm() > epsilon) && (++itr < max_iterations));
  final_param = param;
}

template <typename NUM_TYPE, uint32_t N>
VEC_TYPE NonLinearLSQ<NUM_TYPE, N>::getParam() {
  return final_param;
}

template <typename NUM_TYPE, uint32_t N>
void NonLinearLSQ<NUM_TYPE, N>::setMaxIteration(int val) {}

template <typename NUM_TYPE, uint32_t N>
void NonLinearLSQ<NUM_TYPE, N>::setEpsilon(NUM_TYPE val) {
  epsilon = val;
}

template <typename NUM_TYPE, uint32_t N>
void NonLinearLSQ<NUM_TYPE, N>::setDelta(VEC_TYPE val) {
  delta = val;
}

template <typename NUM_TYPE, uint32_t N>
int NonLinearLSQ<NUM_TYPE, N>::getMaxIteration() {
  return max_iterations;
}

template <typename NUM_TYPE, uint32_t N>
NUM_TYPE NonLinearLSQ<NUM_TYPE, N>::getEpsilon() {
  return epsilon;
}

template <typename NUM_TYPE, uint32_t N>
VEC_TYPE NonLinearLSQ<NUM_TYPE, N>::getDelta() {
  return delta;
}
