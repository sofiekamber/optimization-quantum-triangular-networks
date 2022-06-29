#ifndef DISTRIBUTION_H
#define DISTRIBUTION_H

#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Sparse>

class Distribution
{
public:
    const int M;
    Eigen::VectorXd q_a, q_b, q_c;
    Eigen::VectorXd xi_A, xi_B, xi_C;
    Eigen::VectorXd P = Eigen::VectorXd::Zero(64);;
    // primary constructor
    /**
    * @brief A class to initialize an arbitrary distribution
    * @param M discretization parameter \in {1, ..., 60}
    */
    Distribution(int M_, Eigen::VectorXd q_a, Eigen::VectorXd q_b, Eigen::VectorXd q_c, 
                    Eigen::VectorXd xi_a, Eigen::VectorXd xi_b, Eigen::VectorXd xi_c) : M(M_)
    {
        assert(q_a.size() == M && "Wrong dimension of q_a");
        assert(q_b.size() == M && "Wrong dimension of q_b");
        assert(q_c.size() == M && "Wrong dimension of q_c");
        assert(xi_a.size() == 4 * M * M && "Wrong dimension of xi_a");
        assert(xi_b.size() == 4 * M * M && "Wrong dimension of xi_b");
        assert(xi_c.size() == 4 * M * M && "Wrong dimension of xi_c");
        this->q_a = q_a;
        this->q_b = q_b;
        this->q_c = q_c;
        this->xi_A = xi_a;
        this->xi_B = xi_b;
        this->xi_C = xi_c;
        compute();
    }

    // simplified primary constructor
    Distribution(int M_, Eigen::VectorXd full) : M(M_)
    {
        assert(full.size() == (12 * M * M + 3 * M) && "Wrong input vector dimension");
        q_a = full.segment(0, M);
        q_b = full.segment(M, M);
        q_c = full.segment(2*M, M);
        xi_A = full.segment(3 * M, 4 * M * M);
        xi_B = full.segment(3 * M + 4 * M * M, 4 * M * M);
        xi_C = full.segment(3 * M + 8 * M * M, 4 * M * M);
        compute();

    }

    /**
     * @brief return value of xi_a(a | beta, gamma) \in R^{4M^2}
     *
     * @param a
     * @param beta
     * @param gamma
     * @return double
     */
    double xi_a(int a, int beta, int gamma)
    {
        return xi_A(a * M * M + beta * M + gamma);
    }
    /**
     * @brief return value of xi_b(b | alpha, gamma) \in R^{4M^2}
     *
     * @param b
     * @param alpha
     * @param gamma
     * @return double
     */
    double xi_b(int b, int alpha, int gamma)
    {
        return xi_B(b * M * M + alpha * M + gamma);
    }
    /**
     * @brief return value of xi_c(c | alpha, beta) \in R^{4M^2}
     *
     * @param c
     * @param alpha
     * @param beta
     * @return double
     */
    double xi_c(int c, int alpha, int beta)
    {
        return xi_C(c * M * M + alpha * M + beta);
    }

    bool checkConstraints() {
        return q_a.sum() == 1 && q_b.sum() == 1 && q_c.sum() == 1 && xi_A.sum() == 1 && xi_B.sum() == 1 && xi_C.sum() == 1;
    }


    /**
     * @brief Evaluate distribution P(a, b, c)
     *
     * @param a \in {0, ..., 3}
     * @param b \in {0, ..., 3}
     * @param c \in {0, ..., 3}
     * @return double = P(a, b, c) under intialized distribution
     */

    // put in constructor
    double eval(int a, int b, int c)
    {
        assert(a >= 0 && a < 4 && "Invalid parameter a!");
        assert(b >= 0 && b < 4 && "Invalid parameter b!");
        assert(c >= 0 && c < 4 && "Invalid parameter c!");
        double value = 0;
        for (int i = 0; i < M; i++)
        {
            for (int j = 0; j < M; j++)
            {
                for (int k = 0; k < M; k++)
                {
                    value += q_a[i] * q_b[j] * q_c[k] * xi_a(a, j, k) * xi_b(b, i, k) * xi_c(c, i, j);
                }
            }
        }
        P[a * 16 + b * 4 + c] = value;
        return value;
    }

    /**
     * @brief computes the Jacobian of \Gamma = G at point this.Distribution
     *
     * @return DG as a SparseMatrix
     */
    Eigen::SparseMatrix<double> computeJacobian()
    {
        // order of computing derivative: q_a, q_b, q_c, xi_a, xi_b, xi_c
        const int n = M * M * 12 + 3 * M;

        // encodes the <a, b, c> in R^64 vector
        auto val = [](int a, int b, int c)
        {
            return a * 16 + b * 4 + c;
        };

        Eigen::SparseMatrix<double> J(64, n);
        std::vector<Eigen::Triplet<double>> triplets;

        for (int a = 0; a < 4; a++)
        {
            for (int b = 0; b < 4; b++)
            {
                for (int c = 0; c < 4; c++)
                {
                    // computing q_a derivatives
                    for (int alpha = 0; alpha < M; alpha++)
                    {
                        double value = 0;
                        for (int beta = 0; beta < M; beta++)
                        {
                            for (int gamma = 0; gamma < M; gamma++)
                            {
                                value += q_b(beta) * q_c(gamma) * xi_a(a, beta, gamma) * xi_b(b, alpha, gamma) * xi_c(c, alpha, beta);
                            }
                        }
                        triplets.push_back(Eigen::Triplet<double>(val(a, b, c), alpha, value));
                    }

                    // computing q_b derivatives
                    for (int beta = 0; beta < M; beta++)
                    {
                        double value = 0;
                        for (int alpha = 0; alpha < M; alpha++)
                        {
                            for (int gamma = 0; gamma < M; gamma++)
                            {
                                value += q_a(alpha) * q_c(gamma) * xi_a(a, beta, gamma) * xi_b(b, alpha, gamma) * xi_c(c, alpha, beta);
                            }
                        }
                        triplets.push_back(Eigen::Triplet<double>(val(a, b, c), M + beta, value));
                    }

                    // computing q_c derivatives
                    for (int gamma = 0; gamma < M; gamma++)
                    {
                        double value = 0;
                        for (int alpha = 0; alpha < M; alpha++)
                        {
                            for (int beta = 0; beta < M; beta++)
                            {
                                value += q_a(alpha) * q_b(beta) * xi_a(a, beta, gamma) * xi_b(b, alpha, gamma) * xi_c(c, alpha, beta);
                            }
                        }
                        triplets.push_back(Eigen::Triplet<double>(val(a, b, c), M + M + gamma, value));
                    }

                    // computing xi_a derivatives
                    for (int beta = 0; beta < M; beta++)
                    {
                        for (int gamma = 0; gamma < M; gamma++)
                        {
                            double value = 0;
                            for (int alpha = 0; alpha < M; alpha++)
                            {
                                value += q_a(alpha) * q_b(beta) * q_c(gamma) * xi_b(b, alpha, gamma) * xi_c(c, alpha, beta);
                            }
                            triplets.push_back(Eigen::Triplet<double>(val(a, b, c), 3 * M + xi_a(a, beta, gamma), value));
                        }
                    }
                    // computing xi_b derivatives
                    for (int alpha = 0; alpha < M; alpha++)
                    {
                        for (int gamma = 0; gamma < M; gamma++)
                        {
                            double value = 0;
                            for (int beta = 0; beta < M; beta++)
                            {
                                value += q_a(alpha) * q_b(beta) * q_c(gamma) * xi_a(a, beta, gamma) * xi_c(c, alpha, beta);
                            }
                            triplets.push_back(Eigen::Triplet<double>(val(a, b, c), 3 * M + 4 * M * M + xi_b(b, alpha, gamma), value));
                        }
                    }
                    // computing xi_c derivatives
                    for (int beta = 0; beta < M; beta++)
                    {
                        for (int alpha = 0; alpha < M; alpha++)
                        {
                            double value = 0;
                            for (int gamma = 0; gamma < M; gamma++)
                            {
                                value += q_a(alpha) * q_b(beta) * q_c(gamma) * xi_b(b, alpha, gamma) * xi_a(a, beta, gamma);
                            }
                            triplets.push_back(Eigen::Triplet<double>(val(a, b, c), 3 * M + 8 * M * M + xi_c(c, alpha, beta), value));
                        }
                    }
                }
            }
        }

        J.setFromTriplets(triplets.begin(), triplets.end());
        J.makeCompressed();

        // actually J is not very sparse (4*M*M + 3M values are non-zero), but it has a lot of zero blocks
        return J;
    }
private:
    /**
     * @brief Compute the vector P
     * Called by constructor
     */
    void compute()
    {
        for (int a = 0; a < 4; a++){
            for (int b = 0; b < 4; b++){
                for (int c = 0; c < 4; c++){
                    for (int alpha = 0; alpha < M; alpha++){
                        for (int beta = 0; beta < M; beta++){
                            for (int gamma = 0; gamma < M; gamma++){
                                P[a * 16 + b * 4 + c] += q_a(alpha) * q_b(beta) * q_c(gamma) * xi_a(a, beta, gamma) * xi_b(b, alpha, gamma) * xi_c(c, beta, gamma);
                            }
                        }
                    }
                }
            }
        }
    }
};

#endif
