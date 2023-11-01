#include "NonLinearLSQ.h"

NonLinearLSQ::NonLinearLSQ() {}

NonLinearLSQ::~NonLinearLSQ() {}

ERROR_TYPE NonLinearLSQ::derror_dx(int dx_index, const VECTOR_TYPE& input, const std::vector<ERROR_TYPE>& data_line) {
  VECTOR_TYPE input_delta = input;
  input_delta[dx_index] += delta[dx_index];
  return (error(input_delta, data_line) - error(input, data_line)) / delta[dx_index];
}

void NonLinearLSQ::getSystem(const VECTOR_TYPE& input, MATRIX_TYPE& H, VECTOR_TYPE& b) {
  delta.resize(input.rows(), 1);
  for (int i = 0; i < delta.rows(); i++) delta(i) = DEFAULT_DELTA;
  // zerar sistema
  for (int j, i = 0; i < H.rows(); i++) {
    for (j = 0; j < H.cols(); j++) {
      H(i, j) = 0;
    }
    b(i) = 0;
  }
  // varer dados
  MATRIX_TYPE Hs(input.rows(), input.rows());
  VECTOR_TYPE bs(input.rows());
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

void NonLinearLSQ::setErrorFunction(const std::function<ERROR_TYPE(const VECTOR_TYPE&, const std::vector<ERROR_TYPE>&)>& func) {
  error = func;
}

void NonLinearLSQ::setData(const std::vector<std::vector<ERROR_TYPE>>& data_in) {
  data = &data_in;
}

void NonLinearLSQ::solve(const VECTOR_TYPE& input_gess) {
  VECTOR_TYPE param = input_gess;
  MATRIX_TYPE H(param.rows(), param.rows());
  VECTOR_TYPE b(param.rows());
  VECTOR_TYPE delta_param(param.rows());
  int itr = 1;
  do {
    getSystem(param, H, b);
    delta_param = H.colPivHouseholderQr().solve(-b);
    param += delta_param;
  } while ((delta_param.norm() > DEFAULT_EPSILON) && (++itr < DEFAULT_MAX_ITR));
  final_param = param;
}

VECTOR_TYPE NonLinearLSQ::getParam() {
  return final_param;
}