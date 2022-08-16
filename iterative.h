#ifndef ITERATIVE_SOLVER_H
#define ITERATIVE_SOLVER_H

// standart cpp includes
#include <math.h>
#include <memory>

// Eigen includes
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Sparse>

// LP solver includes
#include <CGAL/QP_models.h>
#include <CGAL/QP_functions.h>

#include "distribution.h"

// choose the fastest available solver (float)
#ifdef CGAL_USE_GMP
#include <CGAL/Gmpzf.h>
typedef CGAL::Gmpzf ET;
#else
#include <CGAL/MP_Float.h>
typedef CGAL::MP_Float ET;
#endif

// program and solution types
typedef CGAL::Quadratic_program<double> Program;
typedef CGAL::Quadratic_program_solution<ET> Solution;

namespace Iterative{
    //uniform distribution
    Eigen::VectorXd uniform = Eigen::VectorXd::Constant(1./64., 64);
    Eigen::VectorXd elegant(64);

    /**
     * @brief check whether a \in R^64 is normalized
     * 
     * @param a Vector (discrete distribution) to be verified
     * @param atol absolute tolerance
     */
    void normalized(Eigen::VectorXd a, double atol = 1e-4){
        assert(1.0 - atol < a.sum() && a.sum() < 1.0 + atol && a.size() == 64 && "Normalization is violated or vector size is != 64");
    }

    /**
     * @brief Constructs a simplex matrix corresponding to argmin || Js - b ||_1 with ||s||_1 = 0 [according to q_a, q_b ... xi_c] 
     * (the zero norm condition required so that the step preserves the norm)
     * @param M discretization constant
     * @param J Jacobian
     * @return std::vector<Eigen::Triplet<double>> LP matrix
     */
    Eigen::MatrixXd LPmatrix(int M, const Eigen::SparseMatrix<double>& J){
        int n = 12 * M * M + 4 * M;
        Eigen::MatrixXd LP = J.transpose() * J;
        return LP;
    }

    /**
     * @brief We use simplex method to solve the linearly constrained minimization problem. 
     * Simplex is needed to ensure that the solution is exactly cosntrained. 
     * Alternative would be "punishment term", however this would allow inexact solutions
     * Note: we are minimizing L1 error norm, not L2 as in Gauss-Newton (because Simplex solves linear)
     * @param LP  = J^t * J
     * @param J = Jacobian at point y_k
     * @param F_yk contraints needed for LP (>= F_yk, >= - F_yk)
     * @param y_k previous values of y_k, we need to ensure that y_k - s >= 0 ==> y_k >= s >= -1
     * @return step s Eigen::VectorXd distribution step
     */
    Eigen::VectorXd quadratic_solver(
        int M, 
        const Eigen::MatrixXd& LP, 
        const Eigen::SparseMatrix<double>& J,
        const Eigen::VectorXd& F_yk, 
        const Eigen::VectorXd& y_k)
    {
        const int n = 12 * M * M + 3 * M;
        //total number of variables = n
        assert(F_yk.size() == 64 && y_k.size() == n && "Dimensions of inputs to quadratic solver are wrong!");
        Program lp (CGAL::EQUAL, false, -1.0, false, 1.0);

        // change it to iteration through a sparse matrix
        // initialize the x^t * J^t * J * x minimization objective = L^2 norm
        // note: we define double the value of matrix here (needed for CGAL)
        for (int i = 0; i < M; i++){
            for (int j = 0; j <= i; j++){
                lp.set_d(i, j, 2 * LP(i, j));
            }
        }

        // initialize: - 2 b^t * J * x
        Eigen::VectorXd c = F_yk.transpose() * J;
        for (int i = 0; i < M; i++){
            lp.set_c(i, -2 * c(i));
        }

        // initialize b^t * b
        lp.set_c0(F_yk.squaredNorm());

        // a helper function to access the values of xi_? easier
        auto xi = [&M](int v, int fst, int snd)->int{
            return v * M * M + fst * M * snd;
        };

        unsigned int offset = 0;

        // default values in CGAL are 0s ==> no need to change b

        // now set the normalization constraints for q_a, q_b, q_c
        for (int j = 0; j < M; j++){
            lp.set_a(offset + 0, j, 1.0);
            lp.set_a(offset + 1, M + j, 1.0);
            lp.set_a(offset + 2, 2*M + j, 1.0);
        }
        offset += 3;

        // now set the normalization constraints for xi_a, xi_b, xi_c (three in total)
        for (int l = 0; l < 2; l++){
            for (int i = 0; i < M; i++){
                for (int j = 0; j < M ; j++){
                    lp.set_a(offset + i * M + j, xi(0, i, j), 1.0);
                    lp.set_a(offset + i * M + j, xi(1, i, j), 1.0);
                    lp.set_a(offset + i * M + j, xi(2, i, j), 1.0);
                    lp.set_a(offset + i * M + j, xi(3, i, j), 1.0);
                }
            }
            offset += M*M;
        }

        // setting the values of upper boundary as y_k      
        for (int i = 0; i < n; i++){
            lp.set_u(i, y_k(i));
        }

        // setting the value of the lower boundary as y_k - 1.0
        for (int i = 0; i < n; i++){
            lp.set_l(i, y_k(i) - 1.0);
        }

        Solution sol = CGAL::solve_quadratic_program(lp, ET());
        assert(sol.solves_quadratic_program(lp) && "Solution doesn't work, might be infeasible?");

        std::cout << "Observed objective value is: " << sol.objective_value() << std::endl;

        Eigen::VectorXd solution(n);

        int j = 0;
        for (auto i = sol.variable_numerators_begin(); i != sol.variable_numerators_end(); i++){
            solution[j] = (*i).to_double();
            j++;
        }
        
        return solution;
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

        //Gauss_newton LSE without damping, since for simplex there is no y_k, we use || . ||_2 instead
        const Eigen::SparseMatrix<double>& J = current.computeJacobian();

        //Constrained linear minimization problem with some exact constraints
        const Eigen::MatrixXd& LP = LPmatrix(M, J);

        //Solve using Simplex, since all constraints are linear
        Eigen::VectorXd s = quadratic_solver(M, LP, J, F_yk, y_k);
        Eigen::VectorXd next_p =  y_k - s;

        //initialize a new distribution class
        Distribution next(M, next_p);

        //check if everything went ok
        normalized(next.P);

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
    Eigen::VectorXd solve(const Distribution& initial, const Eigen::VectorXd& goal, int steps = 2, double atol = -1.0e-8){
        Eigen::VectorXd F_next = initial.P;
        Eigen::VectorXd F_prev = initial.P;
        std::shared_ptr<Distribution> next = std::make_shared<Distribution>(initial);
        std::cout << "--------------------------------------" << std::endl;
        std::cout << "Gauss-Newton local minima search: " << std::endl;
        std::cout << "--------------------------------------" << std::endl;
        int counter = 0;
        do{
            counter++;
            next = std::make_shared<Distribution>(gaussNewtonStep(*next, goal));
            F_prev = F_next;
            F_next = next->P;
            std::cout << "Step " << counter << " error: " << (F_next - goal).norm() << "; Step size: " << (F_next - F_prev).norm() << std::endl;
        }while(((F_next - F_prev).norm() > atol) && steps > counter);

        std::cout << "--------------------------------------" << std::endl;
        return F_next;
    }
}

#endif