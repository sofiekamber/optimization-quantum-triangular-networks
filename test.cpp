#include <eigen3/Eigen/Dense>
#include <iostream>
#include "iterative.h"
#include "distribution.h"


Distribution getRandomDistribution(int M) {
    Eigen::VectorXd q_a = Distribution::generate_random_q(M);
    Eigen::VectorXd q_b = Distribution::generate_random_q(M);
    Eigen::VectorXd q_c = Distribution::generate_random_q(M);
    Eigen::VectorXd xi_a = Distribution::generate_random_xi(M);
    Eigen::VectorXd xi_b = Distribution::generate_random_xi(M);
    Eigen::VectorXd xi_c = Distribution::generate_random_xi(M);

    Distribution distribution(M, q_a, q_b, q_c, xi_a, xi_b, xi_c);

    distribution.checkConstraints();

    return distribution;
}

void testJabobian() {
    int M = 2;
    const double TOLERANCE = 1e-3;
    Distribution x_0_Distribution = getRandomDistribution(M);
    Distribution x_Distribution = getRandomDistribution(M);

    const int MAX_ITERATION = 100;
    Eigen::VectorXd x_0 = x_0_Distribution.getAllCoordinates();
    Eigen::VectorXd x = x_Distribution.getAllCoordinates();
    double errorBefore = 1000.0;
    for (int i = 0; i < MAX_ITERATION; i++) {
        Eigen::VectorXd error = x_Distribution.P - (x_0_Distribution.P + (x_0_Distribution.computeJacobian()) * (x - x_0));

        std::cout << "before: " << errorBefore << ", ";
        std::cout << "now: " << error.norm() << std::endl;

        if (error.norm() * 4.0  - errorBefore <= TOLERANCE) {
            std::cout << "error is 4 times smaller than in last step" << std::endl;
        }

        std::cout << std::endl;

//        std::cout << error.norm() * 4.0  - errorBefore << std::endl;
//        assert(error.norm() * 4.0  - errorBefore <= TOLERANCE);

        errorBefore = error.norm();
        x = (x_0 + x) / 2.0;
        x_Distribution = Distribution(M, x);
    }
}

int main(){
    srand((unsigned int) time(0));
    testJabobian();
    return 0;
}

int test_search(){
    return 0;
}

int test_iterative(){
    return 0;
}