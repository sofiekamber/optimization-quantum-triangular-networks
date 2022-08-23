#ifndef NELDER_MELDER_SEARCH_H
#define NELDER_MELDER_SEARCH_H

#include <eigen3/Eigen/Dense>
#include <cfloat>
#include "distribution.h"
#include <iostream>


namespace NelderMeadSearch {

    class NelderMeadSearch {

        /* create the initial simplex */
        std::vector<Distribution> initialize_simplex(Eigen::VectorXd start, int n, int M) {
            std::vector<Distribution> simplex;
            Distribution startDistribution(M, start);
            simplex.push_back(startDistribution);

//          simplex has n + 1 vertices
            for (int i = 1; i <= n; i++) {
                Eigen::VectorXd q_a = Distribution::generate_random_q(M);
                Eigen::VectorXd q_b = Distribution::generate_random_q(M);
                Eigen::VectorXd q_c = Distribution::generate_random_q(M);
                Eigen::VectorXd xi_a = Distribution::generate_random_xi(M);
                Eigen::VectorXd xi_b = Distribution::generate_random_xi(M);
                Eigen::VectorXd xi_c = Distribution::generate_random_xi(M);

                // TODO maybe take a point between new random point an starting point (should satisfy constraint)
                Distribution distribution(M, q_a, q_b, q_c, xi_a, xi_b, xi_c);

                distribution.checkConstraints();

                simplex.push_back(distribution);
            }

            return simplex;

        }

        /* print out the initial values */
        void print_initial_simplex(std::vector<Distribution> simplex, int n) {
            printf("Initial Values\n");
            std::cout << "n: " << n << "\n";
            for (int j = 0; j <= n; j++) {

//                std::cout << "q_a: " << simplex[j].q_a << "\n";
//                std::cout << "q_b: " << simplex[j].q_b << "\n";
//                std::cout << "q_c: " << simplex[j].q_c << "\n";
//                std::cout << "xi_A: " << simplex[j].xi_A << "\n";
//                std::cout << "xi_B: " << simplex[j].xi_B << "\n";
//                std::cout << "xi_C: " << simplex[j].xi_C << "\n";

                std::cout << "evaluated: " << simplex[j].P << "\n";
            }

        }

        double minimizationNorm(Eigen::VectorXd current, Eigen::VectorXd p_0) {
            return (p_0 - current).norm();
        }

        int getIndexOfLargestPoint(std::vector<Distribution> simplex, Eigen::VectorXd p_0, int n) {

            int largestIndex = 0;
            for (int i = 0; i <= n; i++) {
                if (minimizationNorm(simplex[i].P, p_0) > minimizationNorm(simplex[largestIndex].P, p_0)) {
                    largestIndex = i;
                }
            }

            return largestIndex;
        }

        int getIndexOfSmallestPoint(std::vector<Distribution> simplex, Eigen::VectorXd p_0, int n) {

            int smallestIndex = 0;
            for (int i = 0; i <= n; i++) {
                if (minimizationNorm(simplex[i].P, p_0) < minimizationNorm(simplex[smallestIndex].P, p_0)) {
                    smallestIndex = i;
                }
            }

            return smallestIndex;
        }

        int
        getIndexOfSecondLargestPoint(std::vector<Distribution> simplex, Eigen::VectorXd p_0, int largestIndex, int n) {

            int secondLargestIndex = 0;
            for (int i = 0; i <= n; i++) {
                if (minimizationNorm(simplex[i].P, p_0) > minimizationNorm(simplex[secondLargestIndex].P, p_0) &&
                    minimizationNorm(simplex[i].P, p_0) < minimizationNorm(simplex[largestIndex].P, p_0)) {
                    secondLargestIndex = i;
                }
            }

            return secondLargestIndex;
        }

        void shrinkAllPoints(double scale, int indexOfBestPoint, std::vector<Distribution> &simplex) {
            Eigen::VectorXd bestCoordinates = simplex[indexOfBestPoint].getAllCoordinates();
            for (int i = 0; i < simplex.size(); i++) {
                if (i != indexOfBestPoint) {
                    Eigen::VectorXd currentCoordinates = simplex[i].getAllCoordinates();
                    Eigen::VectorXd scaledCoordinates =
                            bestCoordinates + scale * (currentCoordinates - bestCoordinates);
                    simplex[i] = Distribution(simplex[i].M, scaledCoordinates);
                }
            }
        }


        // get centroid of all points of simplex except worst point
        Eigen::VectorXd getCentroid(std::vector<Distribution> simplex, int worstPointIndex, int n) {
            Eigen::VectorXd centroid = Eigen::VectorXd::Zero(n);

            for (int i = 0; i <= n; i++) {
                if (i != worstPointIndex) {
                    centroid += simplex[i].getAllCoordinates();
                }
            }

            centroid /= n;

            return centroid;
        }

    public:
        Eigen::VectorXd getSolution(Distribution distribution, Eigen::VectorXd goal_P) {
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

            int iSmallest = 0;         /* index of vertex with smallest value */
            int iSecondLargest = 0;   /* index of vertex with second largest value */
            int iLargest = 0;         /* index of vertex with largest value */

            Eigen::VectorXd vReflect(n); // coordinates of reflection point
            Eigen::VectorXd vExpansion(n); // coordinates of expansion point
            Eigen::VectorXd vContraction(n); // coordinates of contraction point
            Eigen::VectorXd vCentroid(n); // coordinates of centroid

            /* create the initial simplex */
            std::vector<Distribution> simplex = initialize_simplex(point, n, M);

            const int MAX_ITERATION = 100;

            for (int iteration = 0; iteration < MAX_ITERATION; iteration++) {
                std::cout << "error: " << minimizationNorm(simplex[iSmallest].P, goal_P) << std::endl;

                iSmallest = getIndexOfSmallestPoint(simplex, goal_P, n);
                iLargest = getIndexOfLargestPoint(simplex, goal_P, n);
                iSecondLargest = getIndexOfSecondLargestPoint(simplex, goal_P, iLargest, n);

                std::cout << "iteration " << iteration << " with worst point index " << iLargest << ": ";

                Distribution vSmallest_Distribution = simplex[iSmallest];
                Distribution vLargest_Distribution = simplex[iLargest];
                Distribution vSecondLargest_Distribution = simplex[iSecondLargest];

                // calculate centroid
                vCentroid = getCentroid(simplex, iLargest, n);
                Distribution vCentroid_Distribution(M, vCentroid);
                vCentroid_Distribution.checkConstraints();

                // reflect largest point on centroid
                vReflect = vCentroid + 0.4 * (vCentroid - simplex[iLargest].getAllCoordinates());
                Distribution vReflect_Distribution = Distribution(M, vReflect);

                if (vReflect_Distribution.satisfiesConstraints()) {
                    // if vReflect is smaller than the second largest point and larger than the smallest, replace largest by vReflect
                    // else if vReflect is smaller than smallest, do expand the reflection point
                    if (minimizationNorm(vReflect_Distribution.P, goal_P) <
                        minimizationNorm(vSecondLargest_Distribution.P, goal_P)
                        && minimizationNorm(vReflect_Distribution.P, goal_P) >
                           minimizationNorm(vSmallest_Distribution.P, goal_P)) {
                        simplex[iLargest] = vReflect_Distribution;
                        std::cout << "Reflection" << std::endl;
                        continue;
                    } else if (minimizationNorm(vReflect_Distribution.P, goal_P) <=
                               minimizationNorm(vSmallest_Distribution.P, goal_P)) {
                        vExpansion = vCentroid + 1.5 * (vReflect - vCentroid);
                        Distribution vExpansion_Distribution = Distribution(M, vExpansion);

                        if (vExpansion_Distribution.satisfiesConstraints()) {
                            if (minimizationNorm(vExpansion_Distribution.P, goal_P) <
                                minimizationNorm(vReflect_Distribution.P, goal_P)) {
                                simplex[iLargest] = vExpansion_Distribution;
                                std::cout << "Expansion" << std::endl;
                                continue;
                            } else {
                                simplex[iLargest] = vReflect_Distribution;
                                std::cout << "Reflection because Expansion is not better than reflection" << std::endl;
                                continue;
                            }

                        } else { // vExpansion does not satisfy constraint, just use reflection
                            simplex[iLargest] = vReflect_Distribution;
                            std::cout << "Reflection because Expansion does not satisfy constraint" << std::endl;
                            continue;
                        }
                    } else { // reflection point is not better than second worst, use contraction
                        // reflection is better than worst
                        if (minimizationNorm(vReflect_Distribution.P, goal_P) <
                            minimizationNorm(vLargest_Distribution.P, goal_P)) {
                            vContraction = vCentroid + 0.5 * (vReflect - vCentroid);
                            Distribution vContraction_Distribution = Distribution(M, vContraction);
                            vContraction_Distribution.checkConstraints(); // contraction point should always satisfy constraints

                            // if contraction point is better than reflection point, replace worst with contraction point
                            if (minimizationNorm(vContraction_Distribution.P, goal_P) <
                                minimizationNorm(vReflect_Distribution.P, goal_P)) {
                                simplex[iLargest] = vContraction_Distribution;
                                std::cout << "Contraction outside" << std::endl;
                                continue;
                            } else {
                                // shrink
                                shrinkAllPoints(0.5, iSmallest, simplex);
                                std::cout << "Shrink" << std::endl;
                                continue;
                            }
                        } else { // reflection is worse or equal than worst
                            vContraction =
                                    vCentroid + 0.5 * (vLargest_Distribution.getAllCoordinates() - vCentroid);
                            Distribution vContraction_Distribution = Distribution(M, vContraction);
                            vContraction_Distribution.checkConstraints();

                            if (minimizationNorm(vContraction_Distribution.P, goal_P) <
                                minimizationNorm(vLargest_Distribution.P, goal_P)) {
                                simplex[iLargest] = vContraction_Distribution;
                                std::cout << "Contraction inside" << std::endl;
                                continue;
                            } else {
                                // shrink
                                shrinkAllPoints(0.5, iSmallest, simplex);
                                std::cout << "Shrink" << std::endl;
                                continue;
                            }
                        }
                    }
                } else { // vReflect does not satisfy constraint
                    vContraction = vCentroid + 0.5 * (vLargest_Distribution.getAllCoordinates() - vCentroid);
                    Distribution vContraction_Distribution = Distribution(M, vContraction);
                    if (minimizationNorm(vContraction_Distribution.P, goal_P) <
                        minimizationNorm(vLargest_Distribution.P, goal_P)) {
                        simplex[iSmallest] = vContraction_Distribution;
                        std::cout << "Contraction inside because reflection does not satisfy constraints" << std::endl;
                        continue;
                    } else {
                        // shrink
                        shrinkAllPoints(0.5, iSmallest, simplex);
                        std::cout << "Shrink" << std::endl;
                        continue;
                    }
                }

            }

            return simplex[iSmallest].P;
//            return Eigen::VectorXd::Zero(n);

        }


    };

}

#endif