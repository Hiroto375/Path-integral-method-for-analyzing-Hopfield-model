#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <Eigen/Dense>
#include <omp.h>

using namespace std;
using namespace Eigen;

// Hopfieldモデル
MatrixXd Hopfield(int N, int T, const MatrixXd& memories, const VectorXd& initial_state) {
    MatrixXd X(T + 1, N);
    X.row(0) = initial_state;

    MatrixXd J = (1.0 / N) * (memories.transpose() * memories);
    for (int i = 0; i < N; i++) J(i, i) = 0.0;

    for (int i = 0; i < T; i++) {
        VectorXd h = J * X.row(i).transpose();
        for (int j = 0; j < N; j++) {
            X(i + 1, j) = (h[j] >= 0) ? 1.0 : -1.0;
        }
    }
    return X;
}

// 1エポック
vector<double> Hopfield_epoch(int N, int T, double alpha, double m0, mt19937& gen) {
    int p = static_cast<int>(alpha * N);

    uniform_real_distribution<double> dist(-1.0, 1.0);
    MatrixXd memories(p, N);
    for (int i = 0; i < p; i++) {
        for (int j = 0; j < N; j++) {
            memories(i, j) = (dist(gen) >= 0) ? 1.0 : -1.0;
        }
    }

    VectorXd initial_state = memories.row(0).transpose();
    int flip_times = static_cast<int>((1 - m0) / 2 * N);

    vector<int> index(N);
    iota(index.begin(), index.end(), 0);
    shuffle(index.begin(), index.end(), gen);

    for (int i = 0; i < flip_times; i++) {
        initial_state(index[i]) *= -1;  // 反転
    }

    MatrixXd X = Hopfield(N, T, memories, initial_state);

    vector<double> m(T + 1);
    VectorXd mem0 = memories.row(0).transpose();
    for (int i = 0; i <= T; i++) {
        m[i] = (X.row(i) * mem0)(0, 0) / N;
    }

    return m;
}

int main() {
    int N = 10000;
    int T = 50;
    double m0 = 0.7;
    int epochs = 20;

    vector<double> Alpha = {0.125, 0.130, 0.135, 0.140, 0.145, 0.150, 0.20, 0.30};

    ofstream fout("Hopfield.csv");
    fout << "alpha,step,mean,std\n";

    for (double alpha : Alpha) {
        vector<vector<double>> all_m(epochs);

        // 並列化してエポックを計算
        #pragma omp parallel for
        for (int e = 0; e < epochs; e++) {
            // 各スレッド用の乱数生成器（シードをユニークにする）
            mt19937 local_gen(1234 + e * 7919 + omp_get_thread_num());
            all_m[e] = Hopfield_epoch(N, T, alpha, m0, local_gen);
        }

        // 各 step で mean, std を計算
        for (int t = 0; t <= T; t++) {
            vector<double> vals(epochs);
            for (int e = 0; e < epochs; e++) {
                vals[e] = all_m[e][t];
            }

            double mean = accumulate(vals.begin(), vals.end(), 0.0) / epochs;
            double sq_sum = 0.0;
            for (double v : vals) sq_sum += (v - mean) * (v - mean);
            double std = sqrt(sq_sum / epochs);

            fout << alpha << "," << t << "," << mean << "," << std << "\n";
        }

        cout << "alpha=" << alpha << " completed.\n";
    }

    fout.close();
    return 0;
}