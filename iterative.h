#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Sparse>
#include "distribution.h"

namespace iterative{

    //idea: to incorporate the x_i > 0 define x_i  = x'_i^2
    Distribution uniform(Eigen::VectorXd::Constant(1./64., 64), 10);

    Distribution gaussNewtonStep(Distribution current){
        auto square = [](double a)->double{
            return a*a;
        };

        auto root = [](double a)->double{
            return sqrt(a);
        };
        
        int M = current.M;
        Eigen::VectorXd point(12 * M * M + 3 * M);

        //initialize the current point in the iteration with q_a, q_b, q_c
        point.segment(0, M) = current.q_a;
        point.segment(M, M) = current.q_b;
        point.segment(2*M, M) = current.q_c;

        //initialize the current point in the iteration with xi_a, xi_b, xi_c
        point.segment(3*M, 4*M*M) = current.xi_a;
        point.segment(3*M + 4*M*M, 4*M*M) = current.xi_b;
        point.segment(3*M + 8*M*M, 4*M*M) = current.xi_c;

        //taking the root to later square it
        point = point.unaryExpr(root);

        //initialize the Jacobian from G(x')
        Eigen::SparseMatrix<double> J;

        //Gauss_newton part

        //Constrained linear minimization problem with some exact constraints

        //Solve using SVD?

        //Square the vector

        //initialize a new distribution class

        //if necessary: compute the error (might be expensive)


    }

    //
    Eigen::VectorXd solve(Distribution initial, Eigen::VectorXd goal){
        return uniform.P;
    }
}