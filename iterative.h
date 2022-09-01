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
#ifndef CGAL_USE_GMP
#include <CGAL/Gmpz.h>
typedef CGAL::Gmpz ET;
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
     * @brief Constructs a simplex matrix corresponding to argmin || Js - b ||_2 with ||s||_1 = 0 [according to q_a, q_b ... xi_c] 
     * (the zero norm condition required so that the step preserves the norm)
     * @param M discretization constant
     * @param J Jacobian
     * @return std::vector<Eigen::Triplet<double>> LP matrix
     */
    Eigen::MatrixXd LPmatrix(int M, const Eigen::SparseMatrix<double>& J){
        int n = 12 * M * M + 4 * M;
        Eigen::SparseMatrix<double> JT = J.transpose();
        Eigen::MatrixXd LP = JT * J;
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
        const Eigen::VectorXd& goal, 
        double epsilon = 1e-7)
    {
        const int n = 12 * M * M + 3 * M;
        //total number of variables = n
        assert(goal.size() == n && "Dimensions of inputs to quadratic solver are wrong!");
        Program lp (CGAL::SMALLER, false, -1.0, false, 1.0);

        /*
        IDEA: improve speed by rewriting the whole thing with cgal matrices -> use a rational approximation for double values
        */
        // initialize the x^t * J^t * J * x minimization objective = L^2 norm
        // note: we define double the value of matrix here (needed for CGAL)
        for (int i = 0; i < n; i++){
            lp.set_d(i, i, 2.0);
        }

        // initialize: - 2 b^t * x
        for (int i = 0; i < n; i++){
            lp.set_c(i, -2 * goal(i));
        }

        // initialize b^t * b
        lp.set_c0(goal.squaredNorm());

        // a helper function to access the values of xi_? easier
        auto xi = [&M](int v, int fst, int snd)->int{
            return v * M * M + fst * M + snd;
        };

        // !NOTE! In CGAL first number is the column number <=> variable number [veery stupid idea]
        // Thus first value is variable, second is constraint (when using set_a)

        int offset = 0; // constraints counter
        int var_offset = 0;

        // now set the normalization constraints for q_a, q_b, q_c
        for (int j = 0; j < M; j++){
            lp.set_a(j, offset + 0, 1.0);
            lp.set_a(M + j, offset + 1, 1.0);
            lp.set_a(2*M + j, offset + 2, 1.0);
        }
        offset += 3;

        for (int j = 0; j < M; j++){
            lp.set_a(j, offset + 0, -1.0);
            lp.set_a(M + j, offset + 1, -1.0);
            lp.set_a(2*M + j, offset + 2, -1.0);
        }
        offset += 3;

        var_offset += 3 * M;

        // we allow tolerance within epsilon for our step: -epsilon < step < epsilon
        // now set the normalization constraints for xi_a, xi_b, xi_c (three in total)
        for (int l = 0; l < 3; l++){
            for (int i = 0; i < M; i++){
                for (int j = 0; j < M ; j++){
                    lp.set_a(var_offset + xi(0, i, j), offset + i * M + j, 1.0);
                    lp.set_a(var_offset + xi(1, i, j), offset + i * M + j, 1.0);
                    lp.set_a(var_offset + xi(2, i, j), offset + i * M + j, 1.0);
                    lp.set_a(var_offset + xi(3, i, j), offset + i * M + j, 1.0);
                }
            }
            offset += M*M;
            var_offset += 4 * M * M;
        }

        var_offset = 3 * M;

        for (int l = 0; l < 3; l++){
            for (int i = 0; i < M; i++){
                for (int j = 0; j < M ; j++){
                    lp.set_a(var_offset + xi(0, i, j), offset + i * M + j, -1.0);
                    lp.set_a(var_offset + xi(1, i, j), offset + i * M + j, -1.0);
                    lp.set_a(var_offset + xi(2, i, j), offset + i * M + j, -1.0);
                    lp.set_a(var_offset + xi(3, i, j), offset + i * M + j, -1.0);
                }
            }
            offset += M*M;
            var_offset += 4 * M * M;
        }

        assert (offset == (6 + 3*M*M + 3*M*M) && "Number of constraints is wrong!");

        // less or equall epsilon norm in each step for normalization
        for (int j = 0; j < 3; j++){
            lp.set_b(j, 1 + epsilon);
        }
        for (int j = 3; j < 6; j++){
            lp.set_b(j, epsilon - 1);
        }
        for (int j = 6; j < 6 + 3 * M * M; j++){
            lp.set_b(j, 1 + epsilon);
        }
        for (int j = 6 + 3 * M * M; j < 6 + 6 * M * M; j++){
            lp.set_b(j, epsilon - 1);
        }

        // setting the values of upper boundary as 1.0     
        for (int i = 0; i < n; i++){
            lp.set_u(i, true, 1.0);
        }

        // setting the value of the lower boundary as 0
        for (int i = 0; i < n; i++){
            lp.set_l(i, true, 0);
        }

        Solution sol = CGAL::solve_nonnegative_quadratic_program(lp, ET());
        // assert(sol.solves_quadratic_program(lp) && "Solution doesn't work, might be infeasible?");

        // std::cout << sol << std::endl;

        Eigen::VectorXd solution(n);

        // CGAL::print_quadratic_program(std::cout, lp, "first_lp");

        int j = 0;
        for (auto i = sol.variable_values_begin(); i != sol.variable_values_end(); i++){
            solution[j] = CGAL::to_double(*i);
            j++;
        }
        return solution;
    }

    Distribution gaussNewtonStep(Distribution& current, Eigen::VectorXd goal, bool& done){
        
        int M = current.M;
        int n = 12 * M * M + 3 * M;
        Eigen::VectorXd y_k(n);

        //initialize the current  y_k in the iteration with q_a, q_b, q_c
        y_k.segment(0, M) = current.q_a;
        y_k.segment(M, M) = current.q_b;
        y_k.segment(2*M, M) = current.q_c;

        // initialize the current  y_k in the iteration with xi_a, xi_b, xi_c
        y_k.segment(3*M, 4*M*M) = current.xi_A;
        y_k.segment(3*M + 4*M*M, 4*M*M) = current.xi_B;
        y_k.segment(3*M + 8*M*M, 4*M*M) = current.xi_C;

        // evaluate F(y_k) with argmin ||F(x)||_2^2 =  argmin ||G(.) - goal||_2^2
        Eigen::VectorXd F_yk = current.P - goal;

        // Gauss_newton LSE without damping, since for simplex there is no y_k, we use || . ||_2 instead
        const Eigen::SparseMatrix<double>& J = current.computeJacobian();
        const Eigen::SparseMatrix<double>& JT = J.transpose();

        // Constrained linear minimization problem with some exact constraints
        // solved via sparse linear least squares

        Eigen::MatrixXd AtA_sparse = (JT * J).toDense();
        
        Eigen::VectorXd F_yk_ext = - JT.toDense() * F_yk;

        Eigen::VectorXd step = AtA_sparse.fullPivLu().solve(F_yk_ext);

        // Verify that the step is useful

        if (std::isnan(step.norm()) || step.norm() < 1e-10 || step.norm() > 1e15){
            std::cout << "Reached a local extremum!" << std::endl;
            done = true;
            return current;
        }

        // We want the steps to be small enought to not walk away from simplex

        step = step / std::max(1.0, step.norm());
        Eigen::VectorXd next_p; 
        Eigen::VectorXd constrained;
        do{
            next_p = y_k + step;

            // project the point on a simplex using LP
            constrained = quadratic_solver(M, next_p);

            //check whether it is improving
            Distribution maybe(M, constrained);

            if ((maybe.P - goal).norm() < (current.P - goal).norm()){
                break;
            }
            
            std::cout << "=== attempted step " << step.norm() << " with error " << (maybe.P - goal).norm() << std::endl;

            // make the step smaller
            step *= 0.5;

        }while (step.norm() > 1e-4);

        //initialize a new distribution class
        Distribution next(M, constrained);

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
    Eigen::VectorXd solve(const Distribution& initial, const Eigen::VectorXd& goal, int steps = 2, double atol = 1.0e-8){
        Eigen::VectorXd F_next = initial.P;
        Eigen::VectorXd F_prev = initial.P;
        bool done = false;
        std::shared_ptr<Distribution> next = std::make_shared<Distribution>(initial);
        std::cout << "--------------------------------------" << std::endl;
        std::cout << "Gauss-Newton local minima search: " << std::endl;
        std::cout << "--------------------------------------" << std::endl;
        int counter = 0;
        do{
            counter++;
            next = std::make_shared<Distribution>(gaussNewtonStep(*next, goal, done));
            F_prev = F_next;
            F_next = next->P;
            std::cout << "Step " << counter << " error: " << (F_next - goal).norm() << "; Step size: " << (F_next - F_prev).norm() << std::endl;
        }while(((F_next - F_prev).norm() > atol) && steps > counter && !done);

        std::cout << "--------------------------------------" << std::endl;
        return next->getAllCoordinates();
    }

}

#endif