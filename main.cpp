#include <iostream>
#include <eigen3/Eigen/Dense>
#include "iterative.h"
#include "search_nelderMead.h"
#include "distribution.h"

// uniform distribution for M = 2
const Distribution uniform(2,
                            Eigen::Vector2d::Constant(2, 0.5),
                            Eigen::Vector2d::Constant(2, 0.5),
                            Eigen::Vector2d::Constant(2, 0.5),
                            Eigen::VectorXd::Constant(16, 1.0/4.0),
                            Eigen::VectorXd::Constant(16, 1.0/4.0),
                            Eigen::VectorXd::Constant(16, 1.0/4.0));

const Distribution uniform_1(1,
                            Eigen::VectorXd::Ones(1),
                            Eigen::VectorXd::Ones(1),
                            Eigen::VectorXd::Ones(1),
                            Eigen::VectorXd::Constant(4, 1.0/4.0),
                            Eigen::VectorXd::Constant(4, 1.0/4.0),
                            Eigen::VectorXd::Constant(4, 1.0/4.0));

const Eigen::VectorXd uniform_vec = Eigen::VectorXd::Constant(64, 1./64.);

Eigen::VectorXd elegantJointDistribution() {
    Eigen::VectorXd p(64);
    for (int a = 0; a < 4; a++) {
        for (int b = 0; b < 4; b++) {
            for (int c = 0; c < 4; c++) {
                double value;
                if (a == b && b == c && a == c) {
                    value = 25.0 / 256.0;
                } else if (a != b && b != c && a != c) {
                    value = 5.0 / 256.0;
                } else {
                    value = 1.0 / 256.0;
                }
                p[a * 16 + b * 4 + c] = value;
            }
        }
    }

    assert(p.sum() == 1.0);

    return p;
}

Distribution generateRandom(int M){
        return Distribution(M,
                            Distribution::generate_random_q(M),
                            Distribution::generate_random_q(M),
                            Distribution::generate_random_q(M),
                            Distribution::generate_random_xi(M),
                            Distribution::generate_random_xi(M),
                            Distribution::generate_random_xi(M));
}



/**
 * @brief Combines GaussNewton for the selection of initial points for the simplex (local minimas)
 *        Uses these points as starting points fro simplex whcih should find their optimum
 * 
 * @param M discretization constant
 * @param goal desired distribution
 */
void gaussNelder(int M, Eigen::VectorXd goal){
    const int n = 12 * M * M + 3 * M;

    // generating starting points using Gauss Newton
    std::vector<Eigen::VectorXd> startingPoints;
    for (int i = 0; i < n + 1; i++){
        Eigen::VectorXd optimum = Iterative::solve(generateRandom(M), goal);
        startingPoints.push_back(optimum);
    }

    // applying MeadNelder on them
    NelderMeadSearch::NelderMeadSearch search;
    search.getBestSolution(generateRandom(M), elegantJointDistribution(), 20, false, startingPoints);
        
}

int main(int argc, char* argv[]) {
    
    bool neadMelder = true, iterative = true, test = false;
    int M = 2; //default

    if (argc > 3){
        std::cout << "invalid number of arguments entered" << std::endl;
        return 0;
    }
    if (argc >= 2){
        if (std::string(argv[1]) == "--nelderMead"){
            iterative = false;
        }
        else if (std::string(argv[1]) == "--gauss"){
            neadMelder = false;
        }
        else if (std::string(argv[1]) == "--test"){
            test = true;
            neadMelder = false;
            iterative = false;
        }
        else{
            std::cout << "Invalid flag, use {--nelderMead, --gauss, --test} or default" << std::endl;
            std::cout << "Received flag: "<< argv[1] << std::endl;
            return 0;
        }
    }
    if (argc == 3){
        M = stoi(std::string(argv[2]));
    }
    std::cout << "Parameter M: " << M << std::endl;

    srand((unsigned int) time(0));


    // just a random distribution
    Eigen::VectorXd random = Eigen::VectorXd::Random(64);
    random = random.cwiseAbs();
    random /= random.sum();

    const Distribution something(2,
                            Distribution::generate_random_q(2),
                            Eigen::Vector2d(0.25, 0.75),
                            Eigen::Vector2d(0.25, 0.75),
                            Distribution::generate_random_xi(2),
                            Eigen::VectorXd::Constant(16, 1.0/4.0),
                            Eigen::VectorXd::Constant(16, 1.0/4.0));


    if (neadMelder){
        NelderMeadSearch::NelderMeadSearch search;
        search.getBestSolution(generateRandom(M), elegantJointDistribution(), 20);
    }


    if (iterative){
        Eigen::VectorXd sol = Iterative::solve(generateRandom(M), elegantJointDistribution(), 10U);       
    //    std::cout << "What we got as an approximation: " << std::endl;
    //    std::cout << sol << std::endl;
    }

    // runs both of them combined
    if (test){
        gaussNelder(M, elegantJointDistribution());
    }

    return 0;
}





