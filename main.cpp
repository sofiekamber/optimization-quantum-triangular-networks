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

const Eigen::VectorXd uniform_vec = Eigen::VectorXd::Constant(64, 1./64.);

int main() {
    srand((unsigned int) time(0));

    const Distribution something(2,
                            Distribution::generate_random_q(2),
                            Eigen::Vector2d(0.25, 0.75),
                            Eigen::Vector2d(0.25, 0.75),
                            Distribution::generate_random_xi(2),
                            Eigen::VectorXd::Constant(16, 1.0/4.0),
                            Eigen::VectorXd::Constant(16, 1.0/4.0));

    const Distribution completelyRandom(2,
                                        Distribution::generate_random_q(2),
                                        Distribution::generate_random_q(2),
                                        Distribution::generate_random_q(2),
                                        Distribution::generate_random_xi(2),
                                        Distribution::generate_random_xi(2),
                                        Distribution::generate_random_xi(2));

    NelderMeadSearch::NelderMeadSearch Sth;
    Sth.getSolution(completelyRandom, uniform_vec);
//
    Eigen::VectorXd sol = Iterative::solve(something, uniform_vec, 3U);
//    NelderMeadSearch::NelderMeadSearch search;
//    Eigen::VectorXd sol2 = search.getSolution(something, uniform_vec);

//    std::cout << "What we got as an approximation: " << std::endl;
//    std::cout << sol << std::endl;

    return 0;
}





