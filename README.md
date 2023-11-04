# Maximum Likelihood Estimation (MLE)
Non Linear Leastsquares solver:
 - Implemented as a templated class in a single header
 - May be used to determine any number of params (given enough data points)
 - Enable use of robust kernels for outlier rejection
 - Allows constraints to curve fitting params
 - Uses paralelism with OMP for faster convergence

Compiled in windows from Visual-Studio-code + cmake with MSVC and msys2. Uses Eigen header only library, included in project.

## Sample
- main.cpp file contains sample data table (y,x) of a noisy gaussian distribution 
- The solver is used to determine the best p's for fitting data to function: y = f(x) = p0 * exp(-pow((x - p1) * p2, 2))

![image](https://github.com/alanff15/MLE/assets/86536099/e797f01b-fc71-433f-8a63-07085576f76b)
