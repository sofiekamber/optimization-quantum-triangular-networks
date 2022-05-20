#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Sparse>
#include "distribution.h"

namespace iterative{

    //idea: to incorporate the x_i > 0 define x_i  = x'_i^2
    Distribution uniform(Eigen::VectorXd::Constant(1./64., 64), 10);

    /**
     * @brief We use simplex method to sole the linearl constrained minimization problem. 
     * Simplex is needed to ensure that the solution is exactly cosntrained. 
     * Alternative would be "punishment term", however this would allow inexact solutions
     * @param A equality constraints
     * @param b rhs of equality constraints
     * @param positive previous values of y_k, we need to ensure that y_k - s > 0
     * @return step s Eigen::VectorXd distribution step
     */
    Eigen::VectorXd simplex(Eigen::MatrixXd A, Eigen::VectorXd rhs, Eigen::VectorXd positive){
        //norm constriants
        //< y_k constraints
        // might not have a solution -> use this minimization of error thing
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
        Distribution next = initial;
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