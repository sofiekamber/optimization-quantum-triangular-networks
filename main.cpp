#include <iostream>
#include <eigen3/Eigen/Dense>
#include "iterative.h"
#include "search.h"
#include "distribution.h"

int main() {
    Eigen::Vector2d a(1, 2);
    std::cout << "Hello, World!" << a << std::endl;
    Eigen::VectorXd uniform(15);
    uniform << Eigen::VectorXd::Constant(3, 1.0), Eigen::VectorXd::Constant(12, 1.0/4.0);
    Distribution un_distr(1, uniform);
    std::cout << un_distr.P << std::endl;
    std::cout << "Computed P!" << std::endl;
    return 0;
}




