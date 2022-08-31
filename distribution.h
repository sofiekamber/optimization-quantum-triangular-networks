#ifndef DISTRIBUTION_H
#define DISTRIBUTION_H

#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Sparse>
#include <cmath>

class Distribution
{
public:
    int M;
    Eigen::VectorXd q_a, q_b, q_c;
    Eigen::VectorXd xi_A, xi_B, xi_C;
    Eigen::VectorXd P = Eigen::VectorXd::Zero(64);

    /**
     * @brief Generate random q stle distributions (needed for testing)
     * 
     * @param M size of the Distribution = M
     * @return Eigen::VectorXd 
     */
    static Eigen::VectorXd generate_random_q(const int M) {
        Eigen::VectorXd q = Eigen::VectorXd::Random(M);
        auto abs_lambda = [](double x)->double{
            return std::abs(x);
        };
        q = q.unaryExpr(abs_lambda);
        return q / q.sum();
    }

    static Eigen::VectorXd generate_random_xi(const int M) {
        Eigen::VectorXd xi = Eigen::VectorXd::Random(4*M*M);
        for (int i = 0; i < M * M; i++){
            double sum = 0;
            for (int j = 0; j < 4; j++){
                sum += std::abs(xi(i + j * M * M));
            }
            for (int j = 0; j < 4; j++){
                xi(i + j * M * M) = std::abs(xi(i + j * M * M)) / sum;
            }
        }
        return xi;
    }

    bool satisfiesConstraints(const double epsilon = 1e-6) const{
        bool qConstraint = q_a.sum() < 1 + epsilon && q_a.sum() > 1.0 - epsilon &&
                q_b.sum() < 1 + epsilon && q_b.sum() > 1.0 - epsilon &&
                q_c.sum() < 1 + epsilon && q_c.sum() > 1.0 - epsilon;

        if (!qConstraint) {
            return false;
        }

        for (int i = 0; i < M*M; i++){
            double sumA = 0.0, sumB = 0.0, sumC = 0.0;
            for (int j = 0; j < 4; j++){
                sumA += std::abs(xi_A(i + j * M * M));
                sumB += std::abs(xi_B(i + j * M * M));
                sumC += std::abs(xi_C(i + j * M * M));
            }
            bool xiConstraint = sumA < 1.0 + epsilon && sumA > 1.0 - epsilon &&
                    sumB < 1.0 + epsilon && sumB > 1.0 - epsilon &&
                    sumC < 1.0 + epsilon && sumC > 1.0 - epsilon;

            if (!xiConstraint) {
                return false;
            }
        }

        return true;
    }

    /**
     * @brief Verify whether the vector is normalized
     * 
     * If vector is not normalized an assert is triggered
     */
    void checkConstraints(const double epsilon = 1e-6) const{
        assert(q_a.sum() < 1 + epsilon && q_a.sum() > 1.0 - epsilon && "q_a is not normalized!");
        assert(q_b.sum() < 1 + epsilon && q_b.sum() > 1.0 - epsilon && "q_b is not normalized!");
        assert(q_c.sum() < 1 + epsilon && q_c.sum() > 1.0 - epsilon && "q_c is not normalized!");
        for (int i = 0; i < M*M; i++){
            double sumA = 0.0, sumB = 0.0, sumC = 0.0;
            for (int j = 0; j < 4; j++){
                sumA += std::abs(xi_A(i + j * M * M));
                sumB += std::abs(xi_B(i + j * M * M));
                sumC += std::abs(xi_C(i + j * M * M));
            }
            assert(sumA < 1.0 + epsilon && sumA > 1.0 - epsilon && "xi_a is not normalized");
            assert(sumB < 1.0 + epsilon && sumB > 1.0 - epsilon && "xi_b is not normalized");
            assert(sumC < 1.0 + epsilon && sumC > 1.0 - epsilon && "xi_c is not noramlized");
        }
    }

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
     * @brief Accessing desired index in the generic xi_* vector xi_(v | fst, snd)
     * 
     * @param v 
     * @param fst 
     * @param snd 
     * @return index in the vector xi
     */
    int xi(int v, int fst, int snd) const{
        return v * M * M + fst * M + snd;
    }

    /**
     * @brief return value of xi_a(a | beta, gamma) \in R^{4M^2}
     *
     * @param a
     * @param beta
     * @param gamma
     * @return double
     */
    double xi_a(int a, int beta, int gamma) const
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
    double xi_b(int b, int alpha, int gamma) const
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
    double xi_c(int c, int alpha, int beta) const
    {
        return xi_C(c * M * M + alpha * M + beta);
    }

    Eigen::VectorXd getAllCoordinates() const{
        Eigen::VectorXd result(12 * M * M + 3 * M);
        result << q_a, q_b, q_c, xi_A, xi_B, xi_C;
        return result;
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
    double eval(int a, int b, int c) const
    {
        assert(a >= 0 && a < 4 && "Invalid parameter a!");
        assert(b >= 0 && b < 4 && "Invalid parameter b!");
        assert(c >= 0 && c < 4 && "Invalid parameter c!");
        
        return P[a * 16 + b * 4 + c];
    }

    /**
     * @brief computes the Jacobian of \Gamma = G at point this.Distribution
     *
     * @return DG as a SparseMatrix
     */
    Eigen::SparseMatrix<double> computeJacobian() const
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
                            triplets.push_back(Eigen::Triplet<double>(val(a, b, c), 3 * M + xi(a, beta, gamma), value));
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
                            triplets.push_back(Eigen::Triplet<double>(val(a, b, c), 3 * M + 4 * M * M + xi(b, alpha, gamma), value));
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
                            triplets.push_back(Eigen::Triplet<double>(val(a, b, c), 3 * M + 8 * M * M + xi(c, alpha, beta), value));
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
                                P[16 * a + 4 * b + c] += q_a(alpha) * q_b(beta) * q_c(gamma) * xi_a(a, beta, gamma) * xi_b(b, alpha, gamma) * xi_c(c, beta, gamma);
                            }
                        }
                    }
                }
            }
        }
    }
};

#endif
