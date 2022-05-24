#include <eigen3/Eigen/Dense>

/**
 * @brief A class to initialize an arbitrary distribution
 * @param M = discretization parameter
 */

class Distribution{
    public:
        const int M;
        Eigen::VectorXd q_a, q_b, q_c;
        Eigen::VectorXd xi_A, xi_B, xi_C;
        Eigen::VectorXd P;
        //primary constructor
        Distribution(int M_, Eigen::VectorXd xi_a, Eigen::VectorXd xi_b, Eigen::VectorXd xi_c,
            Eigen::VectorXd q_a, Eigen::VectorXd q_b, Eigen::VectorXd q_c) : M(M_){
                assert(q_a.size() == M && "Wrong dimension of q_a");
                assert(q_b.size() == M && "Wrong dimension of q_b");
                assert(q_c.size() == M && "Wrong dimension of q_c");
                assert(xi_a.size() == 4*M*M && "Wrong dimension of xi_a");
                assert(xi_b.size() == 4*M*M && "Wrong dimension of xi_b");
                assert(xi_c.size() == 4*M*M && "Wrong dimension of xi_c");
                this->q_a = q_a;
                this->q_b = q_b;
                this->q_c = q_c;
                this->xi_A = xi_a;
                this->xi_B = xi_b;
                this->xi_C = xi_c;
        }

        //simplified primary constructor
        Distribution(int M_, Eigen::VectorXd full) : M(M_){
            assert(full.size() == (4*M*M + 3*M) && "Wrong input vector dimension");
            Distribution(M_, full.segment(0, M),
                    full.segment(M, M),
                    full.segment(2*M, M),
                    full.segment(3*M, 4*M*M),
                    full.segment(3*M + 4*M*M, 4*M*M),
                    full.segment(3*M+8*M*M, 4*M*M)
                    );
        }
        //alternative constructor
        Distribution(Eigen::VectorXd m, int M_) : M (M_){
            assert(m.size() == 64 && "Invalid distribution!");
            P = m;
        }
        
        /**
         * @brief return value of xi_a(a | beta, gamma)
         * 
         * @param a 
         * @param beta 
         * @param gamma 
         * @return double 
         */
        double xi_a(int a, int beta, int gamma){
            return xi_A(gamma * M * M, + beta * M + a);
        }
        /**
         * @brief return value of xi_b(b | alpha, gamma)
         * 
         * @param b 
         * @param alpha 
         * @param gamma 
         * @return double 
         */
        double xi_b(int b, int alpha, int gamma){
            return xi_B(gamma * M * M, + alpha * M + b);
        }
        /**
         * @brief return value of xi_c(c | alpha, beta)
         * 
         * @param c 
         * @param alpha 
         * @param beta 
         * @return double 
         */
        double xi_c(int c, int alpha, int beta){
            return xi_C(beta * M * M, + alpha * M + c);
        }

        /**
         * @brief Compute the vector P
         * 
         */
        void compute(){
            return;
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
                        value += q_a[i] * q_b[j] * q_c[k] * xi_a(a, j, k) * xi_b(b, i, k) * xi_c(c, i, j);
                    }
                }
            }
            P[a * 16 + b * 4 + c] = value;
            return value;
        }
    
};
