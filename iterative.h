#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Sparse>
#include "libalglib/src/optimization.h"
#include "libalglib/src/stdafx.h"
#include <math.h>
#include <memory>
#include "distribution.h"

namespace iterative{
    //uniform distribution
    Eigen::VectorXd uniform = Eigen::VectorXd::Constant(1./64., 64);
    Eigen::VectorXd elegant(64);

    /**
     * @brief check whether a is normalized
     * 
     * @param a Vector (discrete distribution) to be verified
     * @param atol absolute tolerance
     */
    void normalized(Eigen::VectorXd a, double atol = 1e-4){
        assert(1.0 - atol < a.sum() && a.sum() < 1.0 + atol && "Normalization is violated!");
    }

    /**
     * @brief Constructs a simplex matrix corresponding to argmin || Js - b ||_1 with ||s||_1 = 0 [according to q_a, q_b ... xi_c] 
     * (the zero norm condition required so that the step preserves the norm)
     * @param M discretization constant
     * @param J Jacobian
     * @return std::vector<Eigen::Triplet<double>> LP matrix
     */
    Eigen::SparseMatrix<double> LPmatrix(int M, std::vector<Eigen::Triplet<double>> J){
        int n = 12 * M * M + 4 * M;
        Eigen::SparseMatrix<double> LP(64 * 2, n + 64);
        //Goal: u_i = |Js_i - b_i| with i \in {0, ..., 63}
        int rows = 64;
        int cols = n;
        std::vector<Eigen::Triplet<double>> constraints;
        constraints.reserve(2 * (J.size() + 64));

        //initializing upper rows: u_i >= - Js_i + b_i ==> u_i + Js_i >= b_i
        constraints.insert(constraints.begin(), J.begin(), J.end());

        //intializing lower rows: u_i >= Js_i - b_i ==> u_i - Js_i >= - b_i
        for (int i = 0; i < J.size(); i++){
            constraints.push_back(Eigen::Triplet<double>(J[i].row() + 64, J[i].col(), (-1)*J[i].value()));
        }

        //adding those new variables u_i  as 1.0 * u_i
        for (int i = 0; i < 64; i++){
            constraints.push_back(Eigen::Triplet<double>(i, cols + i, 1.0));
            constraints.push_back(Eigen::Triplet<double>(i + 64, cols + i, 1.0));
        }

        LP.setFromTriplets(constraints.begin(), constraints.end());

        return LP;
    }

    /**
     * @brief We use simplex method to solve the linearly constrained minimization problem. 
     * Simplex is needed to ensure that the solution is exactly cosntrained. 
     * Alternative would be "punishment term", however this would allow inexact solutions
     * Note: we are minimizing L1 error norm, not L2 as in Gauss-Newton (because Simplex solves linear)
     * @param LP main inequality constraints
     * @param F_yk contraints needed for LP (>= F_yk, >= - F_yk)
     * @param y_k previous values of y_k, we need to ensure that y_k - s >= 0 ==> y_k >= s >= -1
     * @return step s Eigen::VectorXd distribution step
     */
    Eigen::VectorXd simplex(int M, Eigen::SparseMatrix<double> LP, Eigen::VectorXd F_yk, Eigen::VectorXd y_k){
        int n = 12 * M * M + 4 * M;
        //total number of variables = n + 64

        assert(F_yk.size() == 64 && y_k.size() == n && "Dimensions of inputs to simplex are wrong!");

        //initializing LP solver & the report
        alglib::minlpstate state;
        alglib::minlpreport rep;
        alglib::real_1d_array sol, rhs_u, rhs_l;
        alglib::real_2d_array LP_constr;

        //a global upper bounded on all variables or expressions
        Eigen::VectorXd one = Eigen::VectorXd::Ones(64*2);
        rhs_u.setcontent(one.size(), one.data());

        //creating a solver 
        alglib::minlpcreate(n + 64, state);

        //adding original INEQUALITY constraints as a 2D matrix, note all equations are upper bounded by 1
        Eigen::MatrixXd LP_dense(LP);
        Eigen::VectorXd LP_rhs(64*2);

        LP_rhs << F_yk, (-1) * F_yk;
        LP_constr.setcontent(LP.rows(), LP.cols(), LP_dense.data());
        rhs_l.setcontent(LP_rhs.size(), LP_rhs.data());
        alglib::minlpsetlc2dense(state, LP_constr, rhs_l, rhs_u);

        //adding BOUNDARY conditions for the variables q_a, q_b ... xi_c and u
        Eigen::VectorXd upper(n + 64), lower(n + 64);
        upper << y_k, Eigen::VectorXd::Constant(64, 2.0);
        lower << Eigen::VectorXd::Constant(n, -1.0), Eigen::VectorXd::Zero(64);
        alglib::real_1d_array up_var, low_var;
        up_var.setcontent(n + 64, upper.data());
        low_var.setcontent(n + 64, lower.data());
        alglib::minlpsetbc(state, low_var, up_var);

        //introducing MINIMIZATION objective min (u_0 + u_1 + ... + u_63) a.k.a L1 norm
        Eigen::VectorXd minimize(n + 64);
        minimize << Eigen::VectorXd::Zero(n), Eigen::VectorXd::Ones(64);
        alglib::real_1d_array min;
        min.setcontent(minimize.size(), minimize.data());
        alglib::minlpsetcost(state, min);

        //adding normalization EQUALITY constraints for q_a, q_b, q_c, xi_a, xi_b, xi_c for s (step has a 0 norm)
        Eigen::VectorXd q_a(n+64), q_a(n+64), q_b(n+64), q_c(n+64), xi_a(n+64), xi_b(n+64), xi_c(n+64);
        q_a << Eigen::VectorXd::Ones(M), Eigen::VectorXd::Zero(n + 64 - M);
        q_b << Eigen::VectorXd::Zero(M), Eigen::VectorXd::Ones(M), Eigen::VectorXd::Zero(n + 64 - 2*M);
        q_c << Eigen::VectorXd::Zero(2*M), Eigen::VectorXd::Ones(M), Eigen::VectorXd::Zero(n + 64 - 3*M);
        xi_a << Eigen::VectorXd::Zero(3*M), Eigen::VectorXd::Ones(4*M*M), Eigen::VectorXd::Zero(n + 64 - 3*M - 4*M*M);
        xi_b << Eigen::VectorXd::Zero(3*M + 4*M*M), Eigen::VectorXd::Ones(4*M*M), Eigen::VectorXd::Zero(n + 64 - 3*M - 8*M*M);
        xi_c << Eigen::VectorXd::Zero(3*M + 8*M*M), Eigen::VectorXd::Ones(4*M*M), Eigen::VectorXd::Zero(64);

        alglib::real_1d_array Q_A, Q_B, Q_C, Xi_A, Xi_B, Xi_C;
        Q_A.setcontent(q_a.size(), q_a.data());
        Q_B.setcontent(q_b.size(), q_b.data());
        Q_C.setcontent(q_c.size(), q_c.data());
        Xi_A.setcontent(xi_a.size(), xi_a.data());
        Xi_B.setcontent(xi_b.size(), xi_b.data());
        Xi_C.setcontent(xi_c.size(), xi_c.data());

        alglib::minlpaddlc2dense(state, Q_A, 0.0, 0.0);
        alglib::minlpaddlc2dense(state, Q_B, 0.0, 0.0);
        alglib::minlpaddlc2dense(state, Q_C, 0.0, 0.0);
        alglib::minlpaddlc2dense(state, Xi_A, 0.0, 0.0);
        alglib::minlpaddlc2dense(state, Xi_B, 0.0, 0.0);
        alglib::minlpaddlc2dense(state, Xi_C, 0.0, 0.0);
        
        //scaling, all variables are the same
        Eigen::VectorXd longone = Eigen::VectorXd::Ones(n + 64);
        alglib::real_1d_array s;
        s.setcontent(n + 64, longone.data());
        alglib::minlpsetscale(state, s);

        //solving the LP
        alglib::minlpoptimize(state);
        alglib::minlpresults(state, sol, rep);

        assert(rep.terminationtype > 0 && "Simplex failed. Print INFO for more"); 

        //remove the values of u
        Eigen::VectorXd solution(sol.getcontent());

        std::cout <<"Best error achieved by simplex: " << solution.tail(64).sum() << std::endl;

        return solution.head(3*M + 12 * M * M);
    }

    Distribution gaussNewtonStep(Distribution& current, Eigen::VectorXd goal){
        
        int M = current.M;
        int n = 12 * M * M + 3 * M;
        Eigen::VectorXd y_k(n);

        //initialize the current  y_k in the iteration with q_a, q_b, q_c
        y_k.segment(0, M) = current.q_a;
        y_k.segment(M, M) = current.q_b;
        y_k.segment(2*M, M) = current.q_c;

        //initialize the current  y_k in the iteration with xi_a, xi_b, xi_c
        y_k.segment(3*M, 4*M*M) = current.xi_A;
        y_k.segment(3*M + 4*M*M, 4*M*M) = current.xi_B;
        y_k.segment(3*M + 8*M*M, 4*M*M) = current.xi_C;

        //evaluate F(y_k) with argmin ||F(x)||_2^2 =  argmin ||G(.) - goal||_2^2
        Eigen::VectorXd F_yk = current.P - goal;

        //Gauss_newton LSE without damping, since for simplex there is no y_k, we use || . ||_1 instead

        std::vector<Eigen::Triplet<double>> J = current.computeJacobian().second;

        //Constrained linear minimization problem with some exact constraints
        Eigen::SparseMatrix<double> LP = LPmatrix(M, J);

        //Solve using Simplex, since all constraints are linear

        Eigen::VectorXd s = simplex(M, LP, F_yk, y_k);
        Eigen::VectorXd next_p =  y_k - s;

        //check if everything went ok
        normalized(next_p);

        //initialize a new distribution class
        Distribution next(M, next_p);

        return next;
    }

    /**
     * @brief perfomres modified Gauss Newton steps until the termination criteria is achieved
     * 
     * @param initial starting point
     * @param goal distribution we want to achieve
     * @param atol absolute tolerance default = 1.0e-8
     * @param rtol relative tolerance default = 1.0e-6
     * @return Eigen::VectorXd 
     */
    Eigen::VectorXd solve(Distribution& initial, Eigen::VectorXd goal, double atol = 1.0e-8, double rtol = 1.0e-6){
        Eigen::VectorXd x_k = initial.P;
        std::shared_ptr<Distribution> next = std::make_shared<Distribution>(initial);
        Eigen::VectorXd s;
        std::cout << "Gauss-Newton local minima search error" << std::endl;
        std::cout << "--------------------------------------" << std::endl;
        do{
            next = std::make_shared<Distribution>(gaussNewtonStep(*next, goal));
            x_k = next->P;
            std::cout << (x_k - goal).norm() << std::endl;
        }while(s.norm() > rtol * x_k.norm() && (s.norm() > atol));

        return x_k;
    }
}