#include <iostream>
#include <eigen3/Eigen/Dense>
#include "iterative.h"
#include "distribution.h"

// uniform distribution for M = 2
const Distribution uniform(2, 
                            Eigen::Vector2d::Constant(2, 0.5),
                            Eigen::Vector2d::Constant(2, 0.5),
                            Eigen::Vector2d::Constant(2, 0.5),
                            Eigen::VectorXd::Constant(16, 1.0/4.0),
                            Eigen::VectorXd::Constant(16, 1.0/4.0),
                            Eigen::VectorXd::Constant(16, 1.0/4.0));

const Distribution something(2, 
                            Distribution::generate_random_q(2),
                            Eigen::Vector2d(0.25, 0.75),
                            Eigen::Vector2d(0.25, 0.75),
                            Distribution::generate_random_xi(2),
                            Eigen::VectorXd::Constant(16, 1.0/4.0),
                            Eigen::VectorXd::Constant(16, 1.0/4.0));

const Eigen::VectorXd uniform_vec = Eigen::VectorXd::Constant(64, 1./64.);

int main() {
    std::cout << "Hello, World!" << std::endl;
    std::cout << something.P << std::endl;
    std::cout << "Permuted P!" << std::endl;

    Eigen::VectorXd sol = Iterative::solve(something, uniform_vec, 1U);

    std::cout << "What we got as an approxiamtion" << std::endl;
    std::cout << sol << std::endl;

    return 0;
}





