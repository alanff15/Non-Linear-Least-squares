#pragma once

#include <Eigen/Eigen>

#define DEFAULT_MAX_ITR 500
#define DEFAULT_EPSILON (1e-6)
#define DEFAULT_DELTA (1e-5)

#define VEC_TYPE Eigen::Matrix<NUM_TYPE, N, 1>
#define MAT_TYPE Eigen::Matrix<NUM_TYPE, N, N>

template <typename NUM_TYPE, uint32_t N>
class NonLinearLSQ {
public:
  NonLinearLSQ();
  ~NonLinearLSQ();
  void setErrorFunction(const std::function<NUM_TYPE(const NUM_TYPE[N], const std::vector<NUM_TYPE>&)>& func);
  void setKernel(const std::function<NUM_TYPE(const NUM_TYPE)>& kfunc);
  void setData(const std::vector<std::vector<NUM_TYPE>>& data_in);
  void solve();
  void solve(const NUM_TYPE param_gess[N]);
  void solve(const VEC_TYPE& param_gess);
  VEC_TYPE getParam();
  // set-get
  void setMaxIteration(int val);
  void setEpsilon(NUM_TYPE val);
  void setDelta(VEC_TYPE val);
  void setLimits(int param_index, NUM_TYPE min, NUM_TYPE max);
  int getMaxIteration();
  NUM_TYPE getEpsilon();
  VEC_TYPE getDelta();
  int getLastIterations();
  NUM_TYPE getError();

private:
  NUM_TYPE derror_dx(int dx_index, const VEC_TYPE& param, const std::vector<NUM_TYPE>& data_line);
  void getSystem(const VEC_TYPE& param, MAT_TYPE& H, VEC_TYPE& b);
  // atributos
  int max_iterations;
  int iterations;
  NUM_TYPE epsilon;
  NUM_TYPE last_error;
  VEC_TYPE delta;
  Eigen::Matrix<NUM_TYPE, N, 2> param_limits;
  VEC_TYPE final_param;
  const std::vector<std::vector<NUM_TYPE>>* data;
  std::function<NUM_TYPE(const NUM_TYPE)> kernel;
  std::function<NUM_TYPE(const NUM_TYPE[N], const std::vector<NUM_TYPE>&)> error;
};

///////////////////////////////////////////////////////////////////////////////
// Implementation:

template <typename NUM_TYPE, uint32_t N>
NonLinearLSQ<NUM_TYPE, N>::NonLinearLSQ() {
  max_iterations = DEFAULT_MAX_ITR;
  iterations = 0;
  epsilon = (NUM_TYPE)DEFAULT_EPSILON;
  for (int i = 0; i < delta.rows(); i++) delta(i) = (NUM_TYPE)DEFAULT_DELTA;
  kernel = [](double err) { return 1.0; };
  for (int i = 0; i < param_limits.rows(); i++) {
    param_limits(i, 0) = (NUM_TYPE)(-INFINITY);
    param_limits(i, 1) = (NUM_TYPE)(INFINITY);
  }
}

template <typename NUM_TYPE, uint32_t N>
NonLinearLSQ<NUM_TYPE, N>::~NonLinearLSQ() {}

template <typename NUM_TYPE, uint32_t N>
inline NUM_TYPE NonLinearLSQ<NUM_TYPE, N>::derror_dx(int dx_index, const VEC_TYPE& param, const std::vector<NUM_TYPE>& data_line) {
  VEC_TYPE input_delta = param;
  input_delta[dx_index] += delta[dx_index];
  return (error(&input_delta[0], data_line) - error(&param[0], data_line)) / delta[dx_index];
}

template <typename NUM_TYPE, uint32_t N>
inline void NonLinearLSQ<NUM_TYPE, N>::getSystem(const VEC_TYPE& param, MAT_TYPE& H, VEC_TYPE& b) {
  // definir quantidade de threads
  int threads = std::min(12, (int)data->size());
  MAT_TYPE* Hs = new MAT_TYPE[threads];
  VEC_TYPE* bs = new VEC_TYPE[threads];
  NUM_TYPE* errs = new NUM_TYPE[threads];
  int* dlimits = new int[threads + 1];
  // varrer dados
  dlimits[0] = 0;
  dlimits[1] = (int)data->size() / threads;
  dlimits[threads] = (int)data->size();
  for (int th = 2; th < threads; th++) dlimits[th] = dlimits[th - 1] + dlimits[1];
#pragma omp parallel for schedule(guided)
  for (int th = 0; th < threads; th++) {
    VEC_TYPE Jt;  // rows:IN_TYPE cols:OUT_TYPE
    NUM_TYPE w, err;
    // zerar
    for (int j, i = 0; i < Hs[th].rows(); i++) {
      for (j = 0; j < Hs[th].cols(); j++) {
        Hs[th](i, j) = 0;
      }
      bs[th](i) = 0;
    }
    errs[th] = 0;
    // calcular sistema parcial
    for (int i = dlimits[th]; i < dlimits[th + 1]; i++) {
      err = error(&param[0], (*data)[i]);
      w = kernel(err);
      for (int j = 0; j < Jt.rows(); j++) Jt(j) = derror_dx(j, param, (*data)[i]);
      Hs[th] += w * Jt * Jt.transpose();
      bs[th] += w * Jt * err;
      errs[th] += err;
    }
  }
  // sistema final
  for (int j, i = 0; i < H.rows(); i++) {
    for (j = 0; j < H.cols(); j++) {
      H(i, j) = 0;
    }
    b(i) = 0;
  }
  last_error = 0;
  for (int th = 0; th < threads; th++) {
    H += Hs[th];
    b += bs[th];
    last_error += errs[th];
  }
  delete[] dlimits;
  delete[] Hs;
  delete[] bs;
  delete[] errs;
}

template <typename NUM_TYPE, uint32_t N>
void NonLinearLSQ<NUM_TYPE, N>::setErrorFunction(const std::function<NUM_TYPE(const NUM_TYPE[N], const std::vector<NUM_TYPE>&)>& func) {
  error = func;
  iterations = 0;
}

template <typename NUM_TYPE, uint32_t N>
void NonLinearLSQ<NUM_TYPE, N>::setKernel(const std::function<NUM_TYPE(const NUM_TYPE)>& kfunc) {
  kernel = kfunc;
  iterations = 0;
}

template <typename NUM_TYPE, uint32_t N>
void NonLinearLSQ<NUM_TYPE, N>::setData(const std::vector<std::vector<NUM_TYPE>>& data_in) {
  data = &data_in;
  iterations = 0;
}

template <typename NUM_TYPE, uint32_t N>
void NonLinearLSQ<NUM_TYPE, N>::solve() {
  VEC_TYPE gess = VEC_TYPE::Zero();
  for (int i = 0; i < param_limits.rows(); i++) {
    if (param_limits(i, 0) > (NUM_TYPE)(-INFINITY) && param_limits(i, 1) < (NUM_TYPE)(INFINITY)) {
      gess(i) = (param_limits(i, 0) + param_limits(i, 1)) / (NUM_TYPE)2.0;
    }
  }
  solve(gess);
}

template <typename NUM_TYPE, uint32_t N>
void NonLinearLSQ<NUM_TYPE, N>::solve(const NUM_TYPE param_gess[N]) {
  VEC_TYPE gess;
  for (int i = 0; i < gess.rows(); i++) gess(i) = param_gess[i];
  solve(gess);
}

template <typename NUM_TYPE, uint32_t N>
void NonLinearLSQ<NUM_TYPE, N>::solve(const VEC_TYPE& param_gess) {
  VEC_TYPE param = param_gess;
  MAT_TYPE H;
  VEC_TYPE b;
  VEC_TYPE delta_param;
  int itr = 1;
  // check limits
  for (int i = 0; i < param_limits.rows(); i++) {
    param(i) = (param(i) < param_limits(i, 0) ? param_limits(i, 0) : param(i));
    param(i) = (param(i) > param_limits(i, 1) ? param_limits(i, 1) : param(i));
  }
  do {
    getSystem(param, H, b);
    delta_param = H.colPivHouseholderQr().solve(-b);
    param += delta_param;
    // check limits
    for (int i = 0; i < param_limits.rows(); i++) {
      param(i) = (param(i) < param_limits(i, 0) ? param_limits(i, 0) : param(i));
      param(i) = (param(i) > param_limits(i, 1) ? param_limits(i, 1) : param(i));
    }
  } while ((delta_param.norm() > epsilon) && (++itr < max_iterations));
  iterations = itr;
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
void NonLinearLSQ<NUM_TYPE, N>::setLimits(int param_index, NUM_TYPE min, NUM_TYPE max) {
  param_limits(param_index, 0) = min;
  param_limits(param_index, 1) = max;
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

template <typename NUM_TYPE, uint32_t N>
int NonLinearLSQ<NUM_TYPE, N>::getLastIterations() {
  return iterations;
}

template <typename NUM_TYPE, uint32_t N>
NUM_TYPE NonLinearLSQ<NUM_TYPE, N>::getError() {
  return last_error;
}