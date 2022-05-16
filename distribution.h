#include <eigen3/Eigen/Dense>

/**
 * @brief A class to initialize an arbitrary distribution
 * @param M = discretization parameter
 */

class Distribution{
    public:
        const int M;
        Eigen::VectorXd q_a, q_b, q_c;
        Eigen::VectorXd xi_a, xi_b, xi_c;
        Eigen::VectorXd P;
        //primary constructor
        Distribution(int M_, Eigen::VectorXd xi_a, Eigen::VectorXd xi_b, Eigen::VectorXd xi_c,
            Eigen::VectorXd q_a, Eigen::VectorXd q_b, Eigen::VectorXd q_c) : M(M_){
                assert(q_a.size() == M && "Wrong dimension of q_a");
                assert(q_b.size() == M && "Wrong dimension of q_b");
                assert(q_c.size() == M && "Wrong dimension of q_c");
                assert(xi_a.size() == M && "Wrong dimension of xi_a");
                assert(xi_b.size() == M && "Wrong dimension of xi_b");
                assert(xi_c.size() == M && "Wrong dimension of xi_c");
                this->q_a = q_a;
                this->q_b = q_b;
                this->q_c = q_c;
                this->xi_a = xi_a;
                this->xi_b = xi_b;
                this->xi_c = xi_c;
        }
        //alternative constructor
        Distribution(Eigen::VectorXd m, int M_) : M (M_){
            assert(m.size() == 64 && "Invalid distribution!");
            P = m;
        }
        /**
         * @brief returns the index in a vector xi_ corresponding to xi_(first | second, third)
         * 
         * @param first \in {0, ..., 3}
         * @param second \in {0, ..., M-1}
         * @param third \in {0, ..., M-1}
         * @return int \in {0, ..., 4M^2 - 1}
         */
        int xi_val(int first, int second, int third){
            return first * M * M + second * M + third;
        }

        /**
         * @brief Evaluate distribution P(a, b, c)
         * 
         * @param a \in {0, ..., 3}
         * @param b \in {0, ..., 3}
         * @param c \in {0, ..., 3}
         * @return double = P(a, b, c) under intialized distribution
         */

        //put in constructor
        double eval(int a, int b, int c){
            assert(a >= 0 && a < 4 && "Invalid parameter a!");
            assert(b >= 0 && b < 4 && "Invalid parameter b!");
            assert(c >= 0 && c < 4 && "Invalid parameter c!");
            double value = 0;
            for (int i = 0; i < M; i++){
                for (int j = 0; j < M; j++){
                    for (int k = 0; k < M; k++){
                        value += q_a[i] * q_b[j] * q_c[k] * xi_a[xi_val(a, j, k)] * xi_b[xi_val(b, i, k)] * xi_c[xi_val(c, i, j)];
                    }
                }
            }
            P[(a+1)* (b+1) * (c+1) - 1] = value;
            return value;
        }
    
};
