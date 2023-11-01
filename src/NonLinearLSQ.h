#pragma once

#include <Eigen/Eigen>

#define DEFAULT_MAX_ITR 500
#define DEFAULT_EPSILON (1e-8)
#define DEFAULT_DELTA (1e-5)

template <typename NUM_TYP, uint32_t N>
class NonLinearLSQ {
public:
  NonLinearLSQ();
  ~NonLinearLSQ();

  void setErrorFunction(const std::function<NUM_TYP(const Eigen::Matrix<NUM_TYP, N, 1>&, const std::vector<NUM_TYP>&)>&);
  void setData(const std::vector<std::vector<NUM_TYP>>&);
  void solve(const Eigen::Matrix<NUM_TYP, N, 1>&);
  Eigen::Matrix<NUM_TYP, N, 1> getParam();

private:
  NUM_TYP derror_dx(int, const Eigen::Matrix<NUM_TYP, N, 1>&, const std::vector<NUM_TYP>&);
  void getSystem(const Eigen::Matrix<NUM_TYP, N, 1>&, Eigen::Matrix<NUM_TYP, N, N>&, Eigen::Matrix<NUM_TYP, N, 1>&);

  Eigen::Matrix<NUM_TYP, N, 1> delta;
  Eigen::Matrix<NUM_TYP, N, 1> final_param;
  const std::vector<std::vector<NUM_TYP>>* data;
  std::function<NUM_TYP(const Eigen::Matrix<NUM_TYP, N, 1>&, const std::vector<NUM_TYP>&)> error;
};

///////////////////////////////////////////////////////////////////////////////
//   Implementation:   ////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

template <typename NUM_TYP, uint32_t N>
NonLinearLSQ<NUM_TYP, N>::NonLinearLSQ() {}

template <typename NUM_TYP, uint32_t N>
NonLinearLSQ<NUM_TYP, N>::~NonLinearLSQ() {}

template <typename NUM_TYP, uint32_t N>
NUM_TYP NonLinearLSQ<NUM_TYP, N>::derror_dx(int dx_index, const Eigen::Matrix<NUM_TYP, N, 1>& input, const std::vector<NUM_TYP>& data_line) {
  Eigen::Matrix<NUM_TYP, N, 1> input_delta = input;
  input_delta[dx_index] += delta[dx_index];
  return (error(input_delta, data_line) - error(input, data_line)) / delta[dx_index];
}

template <typename NUM_TYP, uint32_t N>
void NonLinearLSQ<NUM_TYP, N>::getSystem(const Eigen::Matrix<NUM_TYP, N, 1>& input, Eigen::Matrix<NUM_TYP, N, N>& H, Eigen::Matrix<NUM_TYP, N, 1>& b) {
  delta.resize(input.rows(), 1);
  for (int i = 0; i < delta.rows(); i++) delta(i) = (NUM_TYP)DEFAULT_DELTA;
  // zerar sistema
  for (int j, i = 0; i < H.rows(); i++) {
    for (j = 0; j < H.cols(); j++) {
      H(i, j) = 0;
    }
    b(i) = 0;
  }
  // varer dados
  Eigen::Matrix<NUM_TYP, N, N> Hs;
  Eigen::Matrix<NUM_TYP, N, 1> bs;
  Eigen::Matrix<NUM_TYP, N, 1> Jt;  // rows:IN_TYPE cols:OUT_TYPE
  NUM_TYP err;
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

template <typename NUM_TYP, uint32_t N>
void NonLinearLSQ<NUM_TYP, N>::setErrorFunction(const std::function<NUM_TYP(const Eigen::Matrix<NUM_TYP, N, 1>&, const std::vector<NUM_TYP>&)>& func) {
  error = func;
}

template <typename NUM_TYP, uint32_t N>
void NonLinearLSQ<NUM_TYP, N>::setData(const std::vector<std::vector<NUM_TYP>>& data_in) {
  data = &data_in;
}

template <typename NUM_TYP, uint32_t N>
void NonLinearLSQ<NUM_TYP, N>::solve(const Eigen::Matrix<NUM_TYP, N, 1>& input_gess) {
  Eigen::Matrix<NUM_TYP, N, 1> param = input_gess;
  Eigen::Matrix<NUM_TYP, N, N> H;
  Eigen::Matrix<NUM_TYP, N, 1> b;
  Eigen::Matrix<NUM_TYP, N, 1> delta_param;
  int itr = 1;
  do {
    getSystem(param, H, b);
    delta_param = H.colPivHouseholderQr().solve(-b);
    param += delta_param;
  } while ((delta_param.norm() > (NUM_TYP)DEFAULT_EPSILON) && (++itr < DEFAULT_MAX_ITR));
  final_param = param;
}

template <typename NUM_TYP, uint32_t N>
Eigen::Matrix<NUM_TYP, N, 1> NonLinearLSQ<NUM_TYP, N>::getParam() {
  return final_param;
}