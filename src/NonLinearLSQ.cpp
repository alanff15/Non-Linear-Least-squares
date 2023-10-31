#include "NonLinearLSQ.h"

NonLinearLSQ::NonLinearLSQ() {
  for (int j, i = 0; i < delta.rows(); i++) {
    for (j = 0; j < delta.cols(); j++) {
      delta(i, j) = DEFAULT_DELTA;
    }
  }
}

NonLinearLSQ::~NonLinearLSQ() {}

ERROR_TYPE NonLinearLSQ::derror_dx(int dx_index, const PARAM_TYPE& input, const std::vector<ERROR_TYPE>& data_line) {
  PARAM_TYPE input_delta = input;
  input_delta[dx_index] += delta[dx_index];
  return (error(input_delta, data_line) - error(input, data_line)) / delta[dx_index];
}

void NonLinearLSQ::getSystem(const PARAM_TYPE& input, MATRIX_TYPE& H, PARAM_TYPE& b) {
  // zerar sistema
  for (int j, i = 0; i < H.rows(); i++) {
    for (j = 0; j < H.cols(); j++) {
      H(i, j) = 0;
    }
    b(i) = 0;
  }
  // varer dados
  MATRIX_TYPE Hs(input.rows(), input.rows());
  PARAM_TYPE bs;
  MATRIX_TYPE Jt(input.rows(), 1);  // rows:IN_TYPE cols:OUT_TYPE
  ERROR_TYPE err;
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

void NonLinearLSQ::setErrorFunction(const std::function<ERROR_TYPE(const PARAM_TYPE&, const std::vector<ERROR_TYPE>&)>& func) {
  error = func;
}

void NonLinearLSQ::setData(const std::vector<std::vector<ERROR_TYPE>>& data_in) {
  data = &data_in;
}

void NonLinearLSQ::solve(const PARAM_TYPE& input_gess) {
  PARAM_TYPE input = input_gess;
  MATRIX_TYPE H(delta.rows(), delta.rows());
  PARAM_TYPE b;
  PARAM_TYPE delta;
  int itr = 1;
  do {
    getSystem(input, H, b);
    delta = H.colPivHouseholderQr().solve(-b);
    input += delta;
  } while ((delta.norm() > DEFAULT_EPSILON) && (++itr < DEFAULT_MAX_ITR));
  final_param = input;
}

PARAM_TYPE NonLinearLSQ::getParam() {
  return final_param;
}
