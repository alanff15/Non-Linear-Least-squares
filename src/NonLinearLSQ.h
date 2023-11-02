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
  void setErrorFunction(const std::function<NUM_TYPE(const VEC_TYPE&, const std::vector<NUM_TYPE>&)>& func);
  void setData(const std::vector<std::vector<NUM_TYPE>>& data_in);
  void solve(const VEC_TYPE& param_gess);
  VEC_TYPE getParam();
  // set-get
  void setMaxIteration(int val);
  void setEpsilon(NUM_TYPE val);
  void setDelta(VEC_TYPE val);
  int getMaxIteration();
  NUM_TYPE getEpsilon();
  VEC_TYPE getDelta();

private:
  NUM_TYPE derror_dx(int dx_index, const VEC_TYPE& param, const std::vector<NUM_TYPE>& data_line);
  void getSystem(const VEC_TYPE& param, MAT_TYPE& H, VEC_TYPE& b);
  // atributos
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
NUM_TYPE NonLinearLSQ<NUM_TYPE, N>::derror_dx(int dx_index, const VEC_TYPE& param, const std::vector<NUM_TYPE>& data_line) {
  VEC_TYPE input_delta = param;
  input_delta[dx_index] += delta[dx_index];
  return (error(input_delta, data_line) - error(param, data_line)) / delta[dx_index];
}

template <typename NUM_TYPE, uint32_t N>
void NonLinearLSQ<NUM_TYPE, N>::getSystem(const VEC_TYPE& param, MAT_TYPE& H, VEC_TYPE& b) {
  // definir quantidade de threads
  int threads = std::min(12, (int)data->size());
  MAT_TYPE* Hs = new MAT_TYPE[threads];
  VEC_TYPE* bs = new VEC_TYPE[threads];
  int* dlimits = new int[threads + 1];
  // varrer dados
  dlimits[0] = 0;
  dlimits[1] = (int)data->size() / threads;
  dlimits[threads] = (int)data->size();
  for (int th = 2; th < threads; th++) dlimits[th] = dlimits[th - 1] + dlimits[1];
#pragma omp parallel for schedule(guided)
  for (int th = 0; th < threads; th++) {
    VEC_TYPE Jt;  // rows:IN_TYPE cols:OUT_TYPE
    NUM_TYPE err;
    // zerar
    for (int j, i = 0; i < Hs[th].rows(); i++) {
      for (j = 0; j < Hs[th].cols(); j++) {
        Hs[th](i, j) = 0;
      }
      bs[th](i) = 0;
    }
    // calcular sistema parcial
    for (int i = dlimits[th]; i < dlimits[th + 1]; i++) {
      err = error(param, (*data)[i]);
      for (int j = 0; j < Jt.rows(); j++) Jt(j) = derror_dx(j, param, (*data)[i]);
      Hs[th] += Jt * Jt.transpose();
      bs[th] += Jt * err;
    }
  }
  // sistema final
  for (int j, i = 0; i < H.rows(); i++) {
    for (j = 0; j < H.cols(); j++) {
      H(i, j) = 0;
    }
    b(i) = 0;
  }
  for (int th = 0; th < threads; th++) {
    H += Hs[th];
    b += bs[th];
  }
  delete[] dlimits;
  delete[] Hs;
  delete[] bs;
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
void NonLinearLSQ<NUM_TYPE, N>::solve(const VEC_TYPE& param_gess) {
  VEC_TYPE param = param_gess;
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
