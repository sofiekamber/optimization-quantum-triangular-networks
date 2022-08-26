#ifndef NELDER_MELDER_SEARCH_H
#define NELDER_MELDER_SEARCH_H

#include <eigen3/Eigen/Dense>
#include <cfloat>
#include "distribution.h"
#include <iostream>
#include<algorithm>
#include<vector>
#include <random>


namespace NelderMeadSearch {

    class NelderMeadSearch {

        std::vector<Distribution> initializeStructuredSimplex(Eigen::VectorXd start, int n, int M) {
            std::vector<Distribution> simplex;
            Distribution startDistribution(M, start);

            Eigen::MatrixXd identity = Eigen::MatrixXd::Identity(M, M);
            Eigen::MatrixXd identityBig = Eigen::MatrixXd::Identity(M * M, M * M);

            simplex.push_back(startDistribution);

            for (int i = 0; i < M; i++) {
                for (int j = 0; j < M; j++) {
                    for (int k = 0; k < M; k++) {

                        Eigen::VectorXd q_a = identity.col(i);

                        Eigen::VectorXd q_b = identity.col(j);

                        Eigen::VectorXd q_c = identity.col(k);

                        Eigen::VectorXd xi_a(4 * M * M);
                        Eigen::VectorXd xi_b(4 * M * M);
                        Eigen::VectorXd xi_c(4 * M * M);

                        xi_a << Eigen::VectorXd::Constant(M * M, 1.0), Eigen::VectorXd::Zero(3 * M * M);
                        xi_b << Eigen::VectorXd::Constant(M * M, 1.0), Eigen::VectorXd::Zero(3 * M * M);
                        xi_c << Eigen::VectorXd::Constant(M * M, 1.0), Eigen::VectorXd::Zero(3 * M * M);

                        Distribution distribution(M, q_a, q_b, q_c, xi_a, xi_b, xi_c);

                        distribution.checkConstraints();

                        simplex.push_back(distribution);

                    }
                }
            }

            std::cout << "Created " << simplex.size() << "/" << n + 1 << " structured points" << std::endl;

            // fill rest with random points
            for (int i = simplex.size(); i < n + 1; i++) {
                simplex.push_back(getRandomDistribution(M));
            }

            assert(simplex.size() == n + 1);

            return simplex;
        }

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

        /* create the initial simplex */
        std::vector<Distribution> intializeRandomSimplex(Eigen::VectorXd start, int n, int M) {
            std::vector<Distribution> simplex;
            Distribution startDistribution(M, start);
            simplex.push_back(startDistribution);

//          simplex has n + 1 vertices
            for (int i = 1; i <= n; i++) {
                simplex.push_back(getRandomDistribution(M));
            }

            return simplex;

        }

        double minimizationNorm(Eigen::VectorXd current, Eigen::VectorXd p_0) {
            return (p_0 - current).norm();
        }

        void shrinkAllPoints(double scale, std::vector<Distribution> &simplex) {
            Eigen::VectorXd bestCoordinates = simplex[simplex.size() - 1].getAllCoordinates();
            // shrink all except for best
            for (int i = 0; i < simplex.size() - 1; i++) {
                Eigen::VectorXd currentCoordinates = simplex[i].getAllCoordinates();
                Eigen::VectorXd scaledCoordinates =
                        bestCoordinates + scale * (currentCoordinates - bestCoordinates);
                simplex[i] = Distribution(simplex[i].M, scaledCoordinates);

            }
        }

        // get centroid of all points of simplex except worst point
        Eigen::VectorXd getCentroid(std::vector<Distribution> simplex, int n) {
            Eigen::VectorXd centroid = Eigen::VectorXd::Zero(n);

            for (int i = 1; i <= n; i++) {
                centroid += simplex[i].getAllCoordinates();
            }

            centroid /= n;

            return centroid;
        }

        void putNewPointInRightOrder(std::vector<Distribution> &simplex, Eigen::VectorXd goal_P, int n) {
            Distribution newPoint = simplex[0];
            for (int i = 0; i < simplex.size(); i++) {
                if (i == simplex.size() - 1) {
                    simplex.push_back(newPoint);
                    break;
                } else if (minimizationNorm(newPoint.P, goal_P) < minimizationNorm(simplex[i].P, goal_P) &&
                        minimizationNorm(newPoint.P, goal_P) >= minimizationNorm(simplex[i + 1].P, goal_P)) {
                    simplex.insert(simplex.begin() + i + 1, newPoint);

                    break;
                }
            }

            // remove first element
            simplex.erase(simplex.begin());

            assert (simplex.size() == n + 1);
        }

    public:
        Distribution getSolution(Distribution distribution, Eigen::VectorXd goal_P) {
            distribution.checkConstraints();

            int M = distribution.M;
            int n = 12 * M * M + 3 * M;
            Eigen::VectorXd point(n);

            //initialize the current point in the iteration with q_a, q_b, q_c
            point.segment(0, M) = distribution.q_a;
            point.segment(M, M) = distribution.q_b;
            point.segment(2 * M, M) = distribution.q_c;

            //initialize the current point in the iteration with xi_a, xi_b, xi_c
            point.segment(3 * M, 4 * M * M) = distribution.xi_A;
            point.segment(3 * M + 4 * M * M, 4 * M * M) = distribution.xi_B;
            point.segment(3 * M + 8 * M * M, 4 * M * M) = distribution.xi_C;

            Eigen::VectorXd vReflect(n); // coordinates of reflection point
            Eigen::VectorXd vExpansion(n); // coordinates of expansion point
            Eigen::VectorXd vContraction(n); // coordinates of contraction point
            Eigen::VectorXd vCentroid(n); // coordinates of centroid

            double const ALPHA = 1.0; // used for reflection
            double const GAMMA = 1.5; // used for expansion
            double const BETA = 0.5; // used for contraction
            double const DELTA = 0.5; // used for shrinkage

            /* create the initial simplex */
            std::vector<Distribution> simplex = intializeRandomSimplex(point, n, M);
//            std::vector<Distribution> simplex = initializeStructuredSimplex(point, n, M);

            std::cout << "simplex initialized " << simplex.size() << "/" << n << std::endl;


            const int MAX_ITERATION = 6000;

//            std::random_device rd; // obtain a random number from hardware
//            std::mt19937 gen(rd()); // seed the generator
//            std::uniform_int_distribution<> distr(0, simplex.size()-1); // define the range
            sort(simplex.begin(), simplex.end(), [this, goal_P](const Distribution &lhs, const Distribution &rhs) {
                return minimizationNorm(lhs.P, goal_P) > minimizationNorm(rhs.P, goal_P);
            });

            for (int iteration = 0; iteration < MAX_ITERATION; iteration++) {
                // the worst point is the only one that got updated
                putNewPointInRightOrder(simplex, goal_P, n);

                std::cout << "error " << minimizationNorm(simplex[n].P, goal_P) << " in iteration " << iteration
                          << std::endl;

                Distribution vSmallest_Distribution = simplex[n];
                Distribution vLargest_Distribution = simplex[0];
                Distribution vSecondLargest_Distribution = simplex[1];

                // calculate centroid
                vCentroid = getCentroid(simplex, n);
                Distribution vCentroid_Distribution(M, vCentroid);
                vCentroid_Distribution.checkConstraints();

                // reflect largest point on centroid
                vReflect = vCentroid + ALPHA * (vCentroid - simplex[0].getAllCoordinates());

                bool constraintsOk = Distribution(M, vReflect).satisfiesConstraints();
                int counter = 1;
                while (!constraintsOk) {
                    Eigen::VectorXd temp = vCentroid + ALPHA * (vCentroid - simplex[counter].getAllCoordinates());
                    Distribution tempDis = Distribution(M, temp);
                    constraintsOk = tempDis.satisfiesConstraints();
                    if (constraintsOk) {
                        vReflect = temp;
                    }
                    if (counter == n) {
                        std::cout << "terminate" << std::endl;
                        constraintsOk = true;
                    }
                    counter++;
                }

                Distribution vReflect_Distribution = Distribution(M, vReflect);

                if (vReflect_Distribution.satisfiesConstraints()) {
                    // if vReflect is smaller than the second largest point and larger than the smallest, replace largest by vReflect
                    // else if vReflect is smaller than smallest, do expand the reflection point
                    if (minimizationNorm(vReflect_Distribution.P, goal_P) <
                        minimizationNorm(vSecondLargest_Distribution.P, goal_P)
                        && minimizationNorm(vReflect_Distribution.P, goal_P) >=
                           minimizationNorm(vSmallest_Distribution.P, goal_P)) {
                        simplex[0] = vReflect_Distribution;
                        std::cout << "Reflection" << std::endl;
                        continue;
                    } else if (minimizationNorm(vReflect_Distribution.P, goal_P) <
                               minimizationNorm(vSmallest_Distribution.P, goal_P)) {
                        vExpansion = vCentroid + GAMMA * (vReflect - vCentroid);
                        Distribution vExpansion_Distribution = Distribution(M, vExpansion);

                        if (vExpansion_Distribution.satisfiesConstraints()) {
                            if (minimizationNorm(vExpansion_Distribution.P, goal_P) <
                                minimizationNorm(vReflect_Distribution.P, goal_P)) {
                                simplex[0] = vExpansion_Distribution;
                                std::cout << "Expansion" << std::endl;
                                continue;
                            } else {
                                simplex[0] = vReflect_Distribution;
                                std::cout << "Reflection because Expansion is not better than reflection" << std::endl;
                                continue;
                            }

                        } else { // vExpansion does not satisfy constraint, just use reflection
                            simplex[0] = vReflect_Distribution;
                            std::cout << "Reflection because Expansion does not satisfy constraint" << std::endl;
                            continue;
                        }
                    } else { // reflection point is not better than second worst, use contraction
                        // reflection is better than worst
                        if (minimizationNorm(vReflect_Distribution.P, goal_P) <
                            minimizationNorm(vLargest_Distribution.P, goal_P)) {
                            vContraction = vCentroid + BETA * (vReflect - vCentroid);
                            Distribution vContraction_Distribution = Distribution(M, vContraction);
                            vContraction_Distribution.checkConstraints(); // contraction point should always satisfy constraints

                            // if contraction point is better than reflection point, replace worst with contraction point
                            if (minimizationNorm(vContraction_Distribution.P, goal_P) <
                                minimizationNorm(vReflect_Distribution.P, goal_P)) {
                                simplex[0] = vContraction_Distribution;
                                std::cout << "Contraction outside" << std::endl;
                                continue;
                            } else {
                                // shrink
                                shrinkAllPoints(DELTA, simplex);
                                std::cout << "Shrink" << std::endl;
                                break;
                            }
                        } else { // reflection is worse or equal than worst
                            vContraction =
                                    vCentroid + BETA * (vLargest_Distribution.getAllCoordinates() - vCentroid);
                            Distribution vContraction_Distribution = Distribution(M, vContraction);
                            vContraction_Distribution.checkConstraints();

                            if (minimizationNorm(vContraction_Distribution.P, goal_P) <
                                minimizationNorm(vLargest_Distribution.P, goal_P)) {
                                simplex[0] = vContraction_Distribution;
                                std::cout << "Contraction inside" << std::endl;
                                continue;
                            } else {
                                // shrink
                                shrinkAllPoints(DELTA, simplex);
                                std::cout << "Shrink" << std::endl;
                                break;
                            }
                        }
                    }
                } else { // vReflect does not satisfy constraint
                    vContraction = vCentroid + BETA * (vLargest_Distribution.getAllCoordinates() - vCentroid);
                    Distribution vContraction_Distribution = Distribution(M, vContraction);
                    if (minimizationNorm(vContraction_Distribution.P, goal_P) <
                        minimizationNorm(vLargest_Distribution.P, goal_P)) {
                        simplex[0] = vContraction_Distribution;
                        std::cout << "Contraction inside because reflection does not satisfy constraints" << std::endl;
                        continue;
                    } else {
                        // shrink
                        shrinkAllPoints(DELTA, simplex);
                        std::cout << "Shrink" << std::endl;
                        break;
                    }
                }

            }

            return simplex[n];

        }


    };

}

#endif