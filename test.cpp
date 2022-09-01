#include <eigen3/Eigen/Dense>
#include <iostream>
#include "iterative.h"
#include "distribution.h"

const Distribution uniform(2,
                            Eigen::VectorXd::Constant(2, 0.5),
                            Eigen::VectorXd::Constant(2, 0.5),
                            Eigen::VectorXd::Constant(2, 0.5),
                            Eigen::VectorXd::Constant(16, 1.0/4.0),
                            Eigen::VectorXd::Constant(16, 1.0/4.0),
                            Eigen::VectorXd::Constant(16, 1.0/4.0));

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

Distribution getRandomUniform(int M) {
    Eigen::VectorXd q_a = Eigen::VectorXd::Constant(M, 1.0/M);
    Eigen::VectorXd q_b = Eigen::VectorXd::Constant(M, 1.0/M);
    Eigen::VectorXd q_c = Eigen::VectorXd::Constant(M, 1.0/M);

    Eigen::VectorXd xi_a = Eigen::VectorXd::Constant(4 * M * M, 1.0/4.0);
    Eigen::VectorXd xi_b = Eigen::VectorXd::Constant(4 * M * M, 1.0/4.0);
    Eigen::VectorXd xi_c = Eigen::VectorXd::Constant(4 * M * M, 1.0/4.0);

    Distribution distribution(M, q_a, q_b, q_c, xi_a, xi_b, xi_c);

    distribution.checkConstraints();

    return distribution;
}


void testJabobian() {
    int M = 10;
    Distribution x_0_Distribution = getRandomDistribution(M);
    Distribution x_Distribution = getRandomDistribution(M);

    const int MAX_ITERATION = 50;
    Eigen::VectorXd x_0 = x_0_Distribution.getAllCoordinates();
    Eigen::VectorXd x = x_Distribution.getAllCoordinates();
    double errorBefore = 1000.0;
    double TOL = 1e-08;

    Eigen::MatrixXd J = x_0_Distribution.computeJacobian().toDense();

    for (int i = 0; i < MAX_ITERATION; i++) {
        Eigen::VectorXd error = x_Distribution.P - (x_0_Distribution.P + J * (x - x_0));

        std::cout << "before: " << errorBefore << ", ";
        std::cout << "now: " << error.norm() << std::endl;
        std::cout << "Reduction rate: " << errorBefore / error.norm() << std::endl;

        std::cout << std::endl;

        errorBefore = error.norm();
        x = (x_0 + x) / 2.0;
        x_Distribution = Distribution(M, x);

        if (error.norm() < TOL){
            break;
        }
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