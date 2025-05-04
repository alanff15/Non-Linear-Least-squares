#pragma once

#include <Eigen/Eigen>

#define NLLSQ_MAX_ITR 500
#define NLLSQ_EPSILON (1e-6)
#define NLLSQ_DELTA (1e-5)

#define NLLSQ_VEC Eigen::Matrix<NUM_TYPE, PARAM_COUNT, 1>
#define NLLSQ_MAT Eigen::Matrix<NUM_TYPE, PARAM_COUNT, PARAM_COUNT>

template <typename NUM_TYPE, typename DATA_TYPE, uint32_t PARAM_COUNT>
class NonLinearLSQ {
public:
  NonLinearLSQ();
  ~NonLinearLSQ();

  /// @brief Defines error function that relates params with data_lines, must
  // be at minimum when params are correct, preferably has an only global
  // minimum
  /// @param func
  void setErrorFunction(const std::function<NUM_TYPE(const NUM_TYPE params[PARAM_COUNT], const DATA_TYPE& data_line)>& func);

  /// @brief Defines robust kernel function that gives wheights to data_lines
  // based on the error they generate, if not defined the default kernel
  // function always returns 1
  /// @param func
  void setKernel(const std::function<NUM_TYPE(const NUM_TYPE)>& func);

  /// @brief Defines data used for curve fitting
  /// @param data_in
  void setData(const std::vector<DATA_TYPE>& data_in);

  /// @brief Curve fitting, for initial guess of the params it uses the mean
  // value of the limits of each param or zero if limits not defined
  void solve();

  /// @brief Curve fitting, uses 'param_guess' array as initial guess of the
  // params
  /// @param param_guess
  void solve(const NUM_TYPE param_guess[PARAM_COUNT]);

  /// @brief Curve fitting, uses 'param_guess' Eigen::Vector as initial guess
  // of the params
  /// @param param_guess
  void solve(const NLLSQ_VEC& param_guess);

  // set-get:

  /// @brief Returns result of curve fitting
  /// @return Eigen::Vector
  NLLSQ_VEC getParam();

  /// @brief Set max iterations allowd for solution, default value is
  // NLLSQ_MAX_ITR
  /// @param val
  void setMaxIteration(int val);

  /// @brief Set minimum threshold of change in params for considering
  // convergence, default value is NLLSQ_EPSILON
  /// @param val
  void setEpsilon(NUM_TYPE val);

  /// @brief Set increment in params used for computing rate of change
  // numerically, recieves 'val' as an Eigen::Vector
  /// @param val
  void setDelta(NLLSQ_VEC val);

  /// @brief Set limits to param values, helps avoid divergency, if not set the
  // default limits are -INFINITY and INFINITY
  /// @param param_index
  /// @param min
  /// @param max
  void setLimits(int param_index, NUM_TYPE min, NUM_TYPE max);

  /// @brief Returns the maximum number of iterations allowed in a 'solve' call
  /// @return int
  int getMaxIteration();

  /// @brief Returns the actual minimum threshold of change in params for
  // considering convergence
  /// @return Eigen::Vector
  NUM_TYPE getEpsilon();

  /// @brief Returns the actual increment in params used for computing rate of
  // change numerically
  /// @return Eigen::Vector
  NLLSQ_VEC getDelta();

  /// @brief Returns the number of iterations it took to converge in the last
  // 'solve' call
  /// @return int
  int getLastIterations();

  /// @brief Returns the sum of the error function evaluated for each data_line
  /// @return NUM_TYPE
  NUM_TYPE getError();

private:
  NUM_TYPE derror_dx(int dx_index, const NLLSQ_VEC& param, const DATA_TYPE& data_line);
  void getSystem(const NLLSQ_VEC& param, NLLSQ_MAT& H, NLLSQ_VEC& b);
  // atributos
  int max_iterations;
  int iterations;
  NUM_TYPE epsilon;
  NUM_TYPE last_error;
  NLLSQ_VEC delta;
  Eigen::Matrix<NUM_TYPE, PARAM_COUNT, 2> param_limits;
  NLLSQ_VEC final_param;
  const std::vector<DATA_TYPE>* data;
  std::function<NUM_TYPE(const NUM_TYPE)> kernel;
  std::function<NUM_TYPE(const NUM_TYPE[PARAM_COUNT], const DATA_TYPE&)> error;
};

///////////////////////////////////////////////////////////////////////////////
// Implementation:

template <typename NUM_TYPE, typename DATA_TYPE, uint32_t PARAM_COUNT>
NonLinearLSQ<NUM_TYPE, DATA_TYPE, PARAM_COUNT>::NonLinearLSQ() {
  max_iterations = NLLSQ_MAX_ITR;
  iterations = 0;
  epsilon = (NUM_TYPE)NLLSQ_EPSILON;
  for (int i = 0; i < delta.rows(); i++) delta(i) = (NUM_TYPE)NLLSQ_DELTA;
  kernel = [](double err) { return 1.0; };
  for (int i = 0; i < param_limits.rows(); i++) {
    param_limits(i, 0) = (NUM_TYPE)(-INFINITY);
    param_limits(i, 1) = (NUM_TYPE)(INFINITY);
  }
}

template <typename NUM_TYPE, typename DATA_TYPE, uint32_t PARAM_COUNT>
NonLinearLSQ<NUM_TYPE, DATA_TYPE, PARAM_COUNT>::~NonLinearLSQ() {}

template <typename NUM_TYPE, typename DATA_TYPE, uint32_t PARAM_COUNT>
inline NUM_TYPE NonLinearLSQ<NUM_TYPE, DATA_TYPE, PARAM_COUNT>::derror_dx(int dx_index, const NLLSQ_VEC& param, const DATA_TYPE& data_line) {
  NLLSQ_VEC input_delta = param;
  input_delta[dx_index] += delta[dx_index];
  return (error(&input_delta[0], data_line) - error(&param[0], data_line)) / delta[dx_index];
}

template <typename NUM_TYPE, typename DATA_TYPE, uint32_t PARAM_COUNT>
inline void NonLinearLSQ<NUM_TYPE, DATA_TYPE, PARAM_COUNT>::getSystem(const NLLSQ_VEC& param, NLLSQ_MAT& H, NLLSQ_VEC& b) {
  // definir quantidade de threads
  int threads = std::min(12, (int)data->size());
  NLLSQ_MAT* Hs = new NLLSQ_MAT[threads];
  NLLSQ_VEC* bs = new NLLSQ_VEC[threads];
  NUM_TYPE* errs = new NUM_TYPE[threads];
  int* dlimits = new int[threads + 1];
  // varrer dados
  dlimits[0] = 0;
  dlimits[1] = (int)data->size() / threads;
  dlimits[threads] = (int)data->size();
  for (int th = 2; th < threads; th++) dlimits[th] = dlimits[th - 1] + dlimits[1];
#pragma omp parallel for schedule(guided)
  for (int th = 0; th < threads; th++) {
    NLLSQ_VEC Jt;  // rows:IN_TYPE cols:OUT_TYPE
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

template <typename NUM_TYPE, typename DATA_TYPE, uint32_t PARAM_COUNT>
void NonLinearLSQ<NUM_TYPE, DATA_TYPE, PARAM_COUNT>::setErrorFunction(const std::function<NUM_TYPE(const NUM_TYPE[PARAM_COUNT], const DATA_TYPE&)>& func) {
  error = func;
  iterations = 0;
}

template <typename NUM_TYPE, typename DATA_TYPE, uint32_t PARAM_COUNT>
void NonLinearLSQ<NUM_TYPE, DATA_TYPE, PARAM_COUNT>::setKernel(const std::function<NUM_TYPE(const NUM_TYPE)>& kfunc) {
  kernel = kfunc;
  iterations = 0;
}

template <typename NUM_TYPE, typename DATA_TYPE, uint32_t PARAM_COUNT>
void NonLinearLSQ<NUM_TYPE, DATA_TYPE, PARAM_COUNT>::setData(const std::vector<DATA_TYPE>& data_in) {
  data = &data_in;
  iterations = 0;
}

template <typename NUM_TYPE, typename DATA_TYPE, uint32_t PARAM_COUNT>
void NonLinearLSQ<NUM_TYPE, DATA_TYPE, PARAM_COUNT>::solve() {
  NLLSQ_VEC guess = NLLSQ_VEC::Zero();
  for (int i = 0; i < param_limits.rows(); i++) {
    if (param_limits(i, 0) > (NUM_TYPE)(-INFINITY) && param_limits(i, 1) < (NUM_TYPE)(INFINITY)) {
      guess(i) = (param_limits(i, 0) + param_limits(i, 1)) / (NUM_TYPE)2.0;
    }
  }
  solve(guess);
}

template <typename NUM_TYPE, typename DATA_TYPE, uint32_t PARAM_COUNT>
void NonLinearLSQ<NUM_TYPE, DATA_TYPE, PARAM_COUNT>::solve(const NUM_TYPE param_guess[PARAM_COUNT]) {
  NLLSQ_VEC guess;
  for (int i = 0; i < guess.rows(); i++) guess(i) = param_guess[i];
  solve(guess);
}

template <typename NUM_TYPE, typename DATA_TYPE, uint32_t PARAM_COUNT>
void NonLinearLSQ<NUM_TYPE, DATA_TYPE, PARAM_COUNT>::solve(const NLLSQ_VEC& param_guess) {
  NLLSQ_VEC param = param_guess;
  NLLSQ_MAT H;
  NLLSQ_VEC b;
  NLLSQ_VEC delta_param;
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

template <typename NUM_TYPE, typename DATA_TYPE, uint32_t PARAM_COUNT>
NLLSQ_VEC NonLinearLSQ<NUM_TYPE, DATA_TYPE, PARAM_COUNT>::getParam() {
  return final_param;
}

template <typename NUM_TYPE, typename DATA_TYPE, uint32_t PARAM_COUNT>
void NonLinearLSQ<NUM_TYPE, DATA_TYPE, PARAM_COUNT>::setMaxIteration(int val) {
  max_iterations = val;
}

template <typename NUM_TYPE, typename DATA_TYPE, uint32_t PARAM_COUNT>
void NonLinearLSQ<NUM_TYPE, DATA_TYPE, PARAM_COUNT>::setEpsilon(NUM_TYPE val) {
  epsilon = val;
}

template <typename NUM_TYPE, typename DATA_TYPE, uint32_t PARAM_COUNT>
void NonLinearLSQ<NUM_TYPE, DATA_TYPE, PARAM_COUNT>::setDelta(NLLSQ_VEC val) {
  delta = val;
}

template <typename NUM_TYPE, typename DATA_TYPE, uint32_t PARAM_COUNT>
void NonLinearLSQ<NUM_TYPE, DATA_TYPE, PARAM_COUNT>::setLimits(int param_index, NUM_TYPE min, NUM_TYPE max) {
  param_limits(param_index, 0) = min;
  param_limits(param_index, 1) = max;
}

template <typename NUM_TYPE, typename DATA_TYPE, uint32_t PARAM_COUNT>
int NonLinearLSQ<NUM_TYPE, DATA_TYPE, PARAM_COUNT>::getMaxIteration() {
  return max_iterations;
}

template <typename NUM_TYPE, typename DATA_TYPE, uint32_t PARAM_COUNT>
NUM_TYPE NonLinearLSQ<NUM_TYPE, DATA_TYPE, PARAM_COUNT>::getEpsilon() {
  return epsilon;
}

template <typename NUM_TYPE, typename DATA_TYPE, uint32_t PARAM_COUNT>
NLLSQ_VEC NonLinearLSQ<NUM_TYPE, DATA_TYPE, PARAM_COUNT>::getDelta() {
  return delta;
}

template <typename NUM_TYPE, typename DATA_TYPE, uint32_t PARAM_COUNT>
int NonLinearLSQ<NUM_TYPE, DATA_TYPE, PARAM_COUNT>::getLastIterations() {
  return iterations;
}

template <typename NUM_TYPE, typename DATA_TYPE, uint32_t PARAM_COUNT>
NUM_TYPE NonLinearLSQ<NUM_TYPE, DATA_TYPE, PARAM_COUNT>::getError() {
  return last_error;
}