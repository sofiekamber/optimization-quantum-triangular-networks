#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Sparse>
#include "libalglib/src/optimization.h"
#include "libalglib/src/stdafx.h"
#include <math.h>
#include "distribution.h"

namespace iterative{

    Distribution uniform(Eigen::VectorXd::Constant(1./64., 64), 10);

    /**
     * @brief check whether a is normalized
     * 
     * @param a Vector (discrete distribution) to be verified
     * @param atol absolute tolerance
     */
    void normalized(Eigen::VectorXd a, double atol = 1e-5){
        assert(1.0 - atol < a.sum() && a.sum() < 1.0 + atol && "Normalization is violated!");
    }

    std::pair<Eigen::SparseMatrix<double>, std::vector<Eigen::Triplet<double>>> computeJacobian(Distribution A){
        // order of computing derivative: q_a, q_b, q_c, xi_a, xi_b, xi_c
        const int M = A.M;
        const int n = M * M * 12 + 3 * M;
        auto val = [](int a, int b, int c){
            return a * 16 + b * 4 + c;
        };


        Eigen::SparseMatrix<double> J;
        std::vector<Eigen::Triplet<double>> triplets;

        
        for (int a = 0; a < 4; a++){
            for (int b = 0; b < 4; b++){
                for (int c = 0; c < 4; c++){
                    //computing xi_a derivatives
                    for (int beta = 0; beta < M; beta++){
                        for (int gamma = 0; gamma < M; gamma++){
                            double value = 0;
                            for (int alpha = 0; alpha < M; alpha++){
                                value += A.q_a(alpha) * A.q_b(beta) * A.q_c(gamma) * A.xi_b(b, alpha, gamma) * A.xi_c(c, alpha, beta);
                            }
                            triplets.push_back(Eigen::Triplet<double>(val(a, b, c), 3*M + (gamma * M * M + beta * M + a), value));
                        }
                    }
                    //computing xi_b derivatives
                    for (int alpha = 0; alpha < M; alpha++){
                        for (int gamma = 0; gamma < M; gamma++){
                            double value = 0;
                            for (int beta = 0; beta < M; beta++){
                                value += A.q_a(alpha) * A.q_b(beta) * A.q_c(gamma) * A.xi_a(a, beta, gamma) * A.xi_c(c, alpha, beta);
                            }
                            triplets.push_back(Eigen::Triplet<double>(val(a, b, c), 3*M + 4*M*M + (gamma * M * M + alpha * M + b), value));
                        }
                    }
                    //computing xi_c derivatives
                    for (int beta = 0; beta < M; beta++){
                        for (int alpha = 0; alpha < M; alpha++){
                            double value = 0;
                            for (int gamma = 0; gamma < M; gamma++){
                                value += A.q_a(alpha) * A.q_b(beta) * A.q_c(gamma) * A.xi_b(b, alpha, gamma) * A.xi_a(a, beta, gamma);
                            }
                            triplets.push_back(Eigen::Triplet<double>(val(a, b, c), 3*M + 8*M*M + (alpha * M * M + beta * M + c), value));
                        }
                    }
                }
            }
        }

        

        J.setFromTriplets(triplets.begin(), triplets.end());
        J.makeCompressed();

        //actually J is not very sparse (4*M*M + 3M values), but it has a lot of zero blocks
        return std::make_pair(J, triplets);

    }

    /**
     * @brief We use simplex method to sole the linearl constrained minimization problem. 
     * Simplex is needed to ensure that the solution is exactly cosntrained. 
     * Alternative would be "punishment term", however this would allow inexact solutions
     * Note: we are minimizing L1 error norm, not L2 as in Gauss-Newton (because Simplex solves linear)
     * @param A equality constraints
     * @param b rhs of equality constraints
     * @param positive previous values of y_k, we need to ensure that y_k - s > 0
     * @return step s Eigen::VectorXd distribution step
     */
    Eigen::VectorXd simplex(std::vector<Eigen::Triplet<double>> T, int rows, int cols, Eigen::VectorXd rhs, Eigen::VectorXd sth){
        int rows_new = rows;
        int cols_new = cols + 64; 

        Eigen::VectorXd minimize(cols_new);
        minimize << Eigen::VectorXd::Zero(cols), Eigen::VectorXd::Ones(64);

        std::vector<Eigen::Triplet<double>> constraints;
        constraints.reserve(2 * (T.size() + 64));
        constraints.insert(constraints.begin(), T.begin(), T.end());

        //constructing constraint matrix for simplex
        for (int i = 0; i < T.size(); i++){
            constraints.push_back(Eigen::Triplet<double>(T[i].col() + 64, T[i].row(), (-1)*T[i].value()));
        }

        //adding those new variables u with modulo constraint U >= |Tx - b| (modulo of error)
        for (int i = 0; i < 64; i++){
            constraints.push_back(Eigen::Triplet<double>(i, cols + i, 1.0));
            constraints.push_back(Eigen::Triplet<double>(i + 64, cols + i, 1.0));
        }

        //initializing LP solver & the report
        alglib::minlpstate state;
        alglib::minlpreport rep;
        alglib::real_1d_array sol;
        alglib::real_2d_array constr;


        constr.setcontent(A.rows(), A.cols(), constraints.data());
        

        alglib::minlpcreate(cols_new, state);

        //scaling, all variables are the same
        alglib::real_1d_array s = "[1,1]";

        //introducing minimization objective min (u_1 + u_2 + ... + u_64) a.k.a L1 norm
        alglib::real_1d_array min;
        min.setcontent(minimize.size(), minimize.data());
        alglib::minlpsetcost(state, min);

        

        //scaling, all variables are the same
        alglib::real_1d_array s = "[1,1]";
        alglib::minlpsetscale(state, s);
        
        //norm constriants
        //< y_k constraints introduce || optimization similar to L1 norm
        // might not have a solution -> use this minimization of error thing

        //solving the LP
        alglib::minlpoptimize(state);
        alglib::minlpresults(state, sol, rep);

        return rhs;
    }

    Distribution gaussNewtonStep(Distribution& current, Eigen::VectorXd goal, double lambda = 0.01){
        
        int M = current.M;
        int n = 12 * M * M + 3 * M;
        Eigen::VectorXd point(n);

        //initialize the current point in the iteration with q_a, q_b, q_c
        point.segment(0, M) = current.q_a;
        point.segment(M, M) = current.q_b;
        point.segment(2*M, M) = current.q_c;

        //initialize the current point in the iteration with xi_a, xi_b, xi_c
        point.segment(3*M, 4*M*M) = current.xi_A;
        point.segment(3*M + 4*M*M, 4*M*M) = current.xi_B;
        point.segment(3*M + 8*M*M, 4*M*M) = current.xi_C;


        //initialize the Jacobian DF(point)
        Eigen::SparseMatrix<double> J_point;

        //evaluate F(point) with argmin ||F(x)||_2^2 =  argmin ||G(.) - goal||_2^2
        Eigen::VectorXd F_point = current.P - goal;

        //Gauss_newton LSE without damping, since for simplex there is no point, we use || . ||_1 instead

        Eigen::MatrixXd damped = J_point;
        Eigen::MatrixXd rhs =  F_point;

        //Constrained linear minimization problem with some exact constraints
        //Solve using Simplex, since all constraints are linear

        Eigen::VectorXd s = simplex(damped, rhs, point);
        Eigen::VectorXd next_p = point - s;

        //initialize a new distribution class
        Distribution next(M, next_p);

        return next;
    }

    //
    Eigen::VectorXd solve(Distribution initial, Eigen::VectorXd goal, double atol = 1.0e-8, double rtol = 1.0e-6){
        Eigen::VectorXd x_k = initial.P;
        Distribution& next = initial;
        Eigen::VectorXd s;
        std::cout << "Gauss-Newton local minima search error" << std::endl;
        std::cout << "--------------------------------------" << std::endl;
        do{
            next = gaussNewtonStep(next, goal);
            x_k = next.P;
            std::cout << (x_k - goal).norm() << std::endl;
        }while(s.norm() > rtol * x_k.norm() && (s.norm() > atol));

        return x_k;
    }
}