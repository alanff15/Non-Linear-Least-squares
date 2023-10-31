#include <iostream>
#include "NonLinearLSQ.h"

int main() {
  // dados: data[0] = 0.5 * data[1] * exp(-0.1 / data[2])
  std::vector<std::vector<double>> data = {
    {0.215165909, 0.541424337, 0.435451410},   //
    {0.082343417, 0.499607021, 0.090108263},   //
    {0.152911824, 0.356209865, 0.655688484},   //
    {0.151307916, 0.345427797, 0.755746402},   //
    {0.062072589, 0.141048351, 0.783385474},   //
    {0.210911017, 0.538434625, 0.409697399},   //
    {0.097431422, 0.243853868, 0.445884757},   //
    {0.419880274, 0.958346544, 0.757044344},   //
    {0.133205695, 0.468705683, 0.177012036},   //
    {0.000427536, 0.001038838, 0.513685065},   //
    {0.303274068, 0.869445960, 0.277722164}};  //
  // inserir ruido nos dados
  for (auto& d : data) {
    d[0] += (double)(std::rand() % 100 - 50) / 10000.0;
    d[1] += (double)(std::rand() % 100 - 50) / 10000.0;
    d[2] += (double)(std::rand() % 100 - 50) / 10000.0;
  }

  // funcao de erro: y=f(x) -> err=(y-f(x))²
  auto errorFunc = [](const Eigen::Vector2d& param, const std::vector<double>& yx) {
    double ret = yx[0];                                // y
    ret -= param[0] * yx[1] * exp(-param[1] / yx[2]);  // f(x)
    return ret * ret;                                  // err
  };

  // resolver
  NonLinearLSQ lsq;
  lsq.setData(data);
  lsq.setErrorFunction(errorFunc);
  lsq.solve(Eigen::Vector2d(0, 0));

  // mostrar resultado
  std::cout << "Ideal: 0.5\t0.1" << std::endl;
  std::cout << "Calc.: " << lsq.getParam()[0] << "\t" << lsq.getParam()[1] << std::endl;

  return EXIT_SUCCESS;
}