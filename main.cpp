#include <iostream>
#include <eigen3/Eigen/Dense>
#include "iterative.h"
#include "search_nelderMead.h"
#include "distribution.h"

// uniform distribution for M = 2
const Distribution uniform(2,
                            Eigen::Vector2d::Constant(2, 0.5),
                            Eigen::Vector2d::Constant(2, 0.5),
                            Eigen::Vector2d::Constant(2, 0.5),
                            Eigen::VectorXd::Constant(16, 1.0/4.0),
                            Eigen::VectorXd::Constant(16, 1.0/4.0),
                            Eigen::VectorXd::Constant(16, 1.0/4.0));

const Distribution uniform_1(1,
                            Eigen::VectorXd::Ones(1),
                            Eigen::VectorXd::Ones(1),
                            Eigen::VectorXd::Ones(1),
                            Eigen::VectorXd::Constant(4, 1.0/4.0),
                            Eigen::VectorXd::Constant(4, 1.0/4.0),
                            Eigen::VectorXd::Constant(4, 1.0/4.0));

const Eigen::VectorXd uniform_vec = Eigen::VectorXd::Constant(64, 1./64.);

Eigen::VectorXd elegantJointDistribution() {
    Eigen::VectorXd p(64);
    for (int a = 0; a < 4; a++) {
        for (int b = 0; b < 4; b++) {
            for (int c = 0; c < 4; c++) {
                double value;
                if (a == b && b == c && a == c) {
                    value = 25.0 / 256.0;
                } else if (a != b && b != c && a != c) {
                    value = 5.0 / 256.0;
                } else {
                    value = 1.0 / 256.0;
                }
                p[a * 16 + b * 4 + c] = value;
            }
        }
    }

    assert(p.sum() == 1.0);

    return p;
}

int main(int argc, char* argv[]) {
    
    bool neadMelder = true, iterative = true, test = false;
    if (argc > 2){
        std::cout << "invalid number of arguments entered" << std::endl;
        return 0;
    }
    if (argc == 2){
        if (std::string(argv[1]) == "--neadmelder"){
            iterative = false;
        }
        else if (std::string(argv[1]) == "--gauss"){
            neadMelder = false;
        }
        else if (std::string(argv[1]) == "--test"){
            test = true;
            neadMelder = false;
            iterative = false;
        }
        else{
            std::cout << "Invalid flag, use {--neadmelder, --gauss, --test} or default" << std::endl;
            std::cout << "Received flag: "<< argv[1] << std::endl;
            return 0;
        }
    }
    srand((unsigned int) time(0));

    const Distribution something(2,
                            Distribution::generate_random_q(2),
                            Eigen::Vector2d(0.25, 0.75),
                            Eigen::Vector2d(0.25, 0.75),
                            Distribution::generate_random_xi(2),
                            Eigen::VectorXd::Constant(16, 1.0/4.0),
                            Eigen::VectorXd::Constant(16, 1.0/4.0));

    const int myM = 2;
    const Distribution completelyRandom(myM,
                                        Distribution::generate_random_q(myM),
                                        Distribution::generate_random_q(myM),
                                        Distribution::generate_random_q(myM),
                                        Distribution::generate_random_xi(myM),
                                        Distribution::generate_random_xi(myM),
                                        Distribution::generate_random_xi(myM));
    if (neadMelder){
        NelderMeadSearch::NelderMeadSearch search;
        search.getBestSolution(completelyRandom, elegantJointDistribution(), 20);
    }

    if (iterative){
        Eigen::VectorXd sol = Iterative::solve(something, elegantJointDistribution(), 1U);       
    //    std::cout << "What we got as an approximation: " << std::endl;
    //    std::cout << sol << std::endl;
    }


    if (test){
        // TODO
    }

    return 0;
}





