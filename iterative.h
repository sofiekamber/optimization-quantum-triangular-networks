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
    Eigen::VectorXd simplex(Eigen::MatrixXd A, Eigen::VectorXd rhs, Eigen::VectorXd positive){
        int n = A.cols();

        //initializing LP solver & the report
        alglib::minlpstate state;
        alglib::minlpreport rep;
        alglib::real_1d_array sol;
        alglib::minlpcreate(n, state);

        //scaling, all variables are the same
        alglib::real_1d_array s = "[1,1]";
        alglib::minlpsetscale(state, s);

        //minimization constraints of the L1 norm
        alglib::minlpsetcost(state, s);
        
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
        point.segment(3*M, 4*M*M) = current.xi_a;
        point.segment(3*M + 4*M*M, 4*M*M) = current.xi_b;
        point.segment(3*M + 8*M*M, 4*M*M) = current.xi_c;


        //initialize the Jacobian DF(point)
        Eigen::SparseMatrix<double> J_point;

        //evaluate F(point) with argmin ||F(x)||_2^2 =  argmin ||G(.) - goal||_2^2
        Eigen::VectorXd F_point = current.P - goal;

        //Gauss_newton LSE with damping

        Eigen::MatrixXd damped = J_point.transpose() * J_point + lambda * Eigen::MatrixXd::Identity(n, n);
        Eigen::MatrixXd rhs =  - J_point.transpose() * F_point;

        //Constrained linear minimization problem with some exact constraints
        //Solve using Simplex, since all constraints are linear

        Eigen::VectorXd s = simplex(damped, rhs, point);
        Eigen::VectorXd next_p = point - s;

        //initialize a new distribution class
        Distribution next(M, next_p);
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