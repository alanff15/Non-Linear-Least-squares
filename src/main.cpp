#include <iostream>
#include "NonLinearLSQ.h"

int main() {
  // criar solver
  NonLinearLSQ<float, 2> lsq;

  // inserir dados
  std::vector<std::vector<float>> dados = {
    {0.215165909f, 0.541424337f, 0.435451410f},
    {0.082343417f, 0.499607021f, 0.090108263f},
    {0.152911824f, 0.356209865f, 0.655688484f},
    {0.151307916f, 0.345427797f, 0.755746402f},
    {0.062072589f, 0.141048351f, 0.783385474f},
    {0.210911017f, 0.538434625f, 0.409697399f},
    {0.097431422f, 0.243853868f, 0.445884757f},
    {0.419880274f, 0.958346544f, 0.757044344f},
    {0.133205695f, 0.468705683f, 0.177012036f},
    {0.000427536f, 0.001038838f, 0.513685065f},
    {0.303274068f, 0.869445960f, 0.277722164f}};
  lsq.setData(dados);

  // definir funcao de erro: y = f(x) | err = (y-f(x))²
  lsq.setErrorFunction([](const Eigen::Vector2f& param, const std::vector<float>& yx) {
    float ret = yx[0];                                 // y
    ret -= param[0] * yx[1] * exp(-param[1] / yx[2]);  // f(x)
    return ret * ret;                                  // err
  });

  // resolver
  lsq.solve(Eigen::Vector2f(0, 0));

  // mostrar resultado
  std::cout << "Params:  " << lsq.getParam()[0] << "\t" << lsq.getParam()[1] << std::endl;
  std::cout << "Iter.:   " << lsq.getLastIterations() << std::endl;
  std::cout << "Erro:    " << lsq.getError() << std::endl;

  return EXIT_SUCCESS;
}