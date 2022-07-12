#ifndef NELDER_MELDER_SEARCH_H
#define NELDER_MELDER_SEARCH_H

#include <eigen3/Eigen/Dense>
#include <cfloat>
#include "distribution.h"
#include <iostream>


namespace NelderMeadSearch{

    class NelderMeadSearch {

        /* create the initial simplex */
        std::vector<Distribution> initialize_simplex(Eigen::VectorXd start, int n, int M)
        {
            std::vector<Distribution> simplex;

            // simplex has n + 1 vertices
            for (int i = 0; i <= n; i++) {
                Distribution distribution(M, start); // TODO what initial distributions?
                simplex.push_back(distribution);
            }

            return simplex;

        }

        /* print out the initial values */
        void print_initial_simplex(std::vector<Distribution> simplex, int n)
        {
            printf("Initial Values\n");
            std::cout << "n: " << n << "\n";
            for (int j=0;j<=n;j++) {

                std::cout << "q_a: " << simplex[j].q_a << "\n";
                std::cout << "q_b: " << simplex[j].q_b << "\n";
                std::cout << "q_c: " << simplex[j].q_c << "\n";
                std::cout << "xi_A: " << simplex[j].xi_A << "\n";
                std::cout << "xi_B: " << simplex[j].xi_B << "\n";
                std::cout << "xi_C: " << simplex[j].xi_C << "\n";

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

        int getIndexOfSecondLargestPoint(std::vector<Distribution> simplex, Eigen::VectorXd p_0, int largestIndex, int n) {

            int secondLargestIndex = 0;
            for (int i = 0; i <= n; i++) {
                if (minimizationNorm(simplex[i].P, p_0) > minimizationNorm(simplex[secondLargestIndex].P, p_0) &&
                        minimizationNorm(simplex[i].P, p_0) < minimizationNorm(simplex[largestIndex].P, p_0)) {
                    secondLargestIndex = i;
                }
            }

            return secondLargestIndex;
        }

    public:
        Eigen::VectorXd getSolution(Distribution& distribution, Eigen::VectorXd p_0) {
            if (!distribution.checkConstraints()) {
                throw std::invalid_argument("Constraint violated");
            }

            int M = distribution.M;
            int n = 12 * M * M + 3 * M;
            Eigen::VectorXd point(n);

            //initialize the current point in the iteration with q_a, q_b, q_c
            point.segment(0, M) = distribution.q_a;
            point.segment(M, M) = distribution.q_b;
            point.segment(2*M, M) = distribution.q_c;

            //initialize the current point in the iteration with xi_a, xi_b, xi_c
            point.segment(3*M, 4*M*M) = distribution.xi_A;
            point.segment(3*M + 4*M*M, 4*M*M) = distribution.xi_B;
            point.segment(3*M + 8*M*M, 4*M*M) = distribution.xi_C;

            int vs;         /* vertex with smallest value */
            int vh;         /* vertex with next smallest value */
            int vg;         /* vertex with largest value */

            int i,j,row;
            int k;   	  /* track the number of function evaluations */
            int itr;	  /* track the number of iterations */

            Eigen::MatrixXd v(n+1, n); // vertices of simplex
            Eigen::VectorXd f(n+1); // value of function at each vertex

            Eigen::VectorXd vr(n); // coordinates of reflection point
            Eigen::VectorXd ve(n); // coordinates of expansion point
            Eigen::VectorXd vc(n); // coordinates of contraction point
            Eigen::VectorXd vm(n); // coordinates of centroid

            double fr; // value of function at reflection point
            double fe; // value of function at expansion point
            double fc; // value of function at contraction point

            double fsum,favg,s;

            /* create the initial simplex */
            double scale = 1.0e-4;
            std::vector<Distribution> simplex = initialize_simplex(point,n, M);

            /* find the initial function values */
            k = n+1;

            /* print out the initial values */
            print_initial_simplex(simplex,n);

            return point;

        }


    };

}

#endif