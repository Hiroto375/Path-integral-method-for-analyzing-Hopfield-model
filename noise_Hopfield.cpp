#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <numeric>
#include <cmath>
#include <Eigen/Dense>
#include <omp.h>

using namespace std;
using namespace Eigen;

// ノイズ付き Hopfield
MatrixXd NoiseHopfield(int N, int T, MatrixXd &memories, VectorXd &initial_state, double sigma, mt19937 &gen) {
    MatrixXd X(T+1, N);
    X.row(0) = initial_state.transpose();

    // J の作成
    MatrixXd J = (1.0 / N) * (memories.transpose() * memories);
    for (int i = 0; i < N; i++) J(i, i) = 0.0;

    normal_distribution<double> dist(0.0, sqrt(sigma / N));

    // ノイズ追加
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < i; j++) {
            double e = dist(gen);
            J(i, j) += e;
            J(j, i) += e;
        }
    }

    // 状態更新
    for (int t = 0; t < T; t++) {
        VectorXd h = J * X.row(t).transpose();
        for (int j = 0; j < N; j++) {
            X(t+1, j) = (h(j) >= 0 ? 1.0 : -1.0);
        }
    }

    return X;
}

// 1エポック分
vector<double> noise_epoch(int N, int T, double alpha, double m0, double sigma, mt19937 &gen) {
    int p = static_cast<int>(alpha * N);

    // メモリー作成
    uniform_real_distribution<double> dist(-1.0, 1.0);
    MatrixXd memories(p, N);
    for (int i = 0; i < p; i++) {
        for (int j = 0; j < N; j++) {
            memories(i, j) = (dist(gen) >= 0 ? 1.0 : -1.0);
        }
    }

    // 初期状態
    VectorXd initial_state = memories.row(0).transpose();
    int flip_times = static_cast<int>((1 - m0) / 2 * N);
    vector<int> indices(N);
    iota(indices.begin(), indices.end(), 0);
    shuffle(indices.begin(), indices.end(), gen);
    for (int i = 0; i < flip_times; i++) initial_state(indices[i]) *= -1;

    // Hopfield 実行
    MatrixXd X = NoiseHopfield(N, T, memories, initial_state, sigma, gen);

    // m(t) を計算
    vector<double> m(T+1);
    VectorXd mem0 = memories.row(0).transpose();
    for (int t = 0; t <= T; t++) {
        m[t] = X.row(t).dot(mem0) / N;
    }
    return m;
}

int main() {
    int N = 10000;   // ニューロン数
    int T = 50;     // ステップ数
    double m0 = 0.7;
    double sigma = 0.1;
    int epochs = 20;

    vector<double> Alpha = {0.085, 0.090, 0.095, 0.10, 0.105, 0.110, 0.20, 0.30};

    mt19937 gen(12345);

    ofstream csv("noise_Hopfield.csv");
    csv << "alpha,step,mean,std\n";

    for (double alpha : Alpha) {
        vector<vector<double>> all_m(epochs);

        // 並列実行
        #pragma omp parallel for
        for (int e = 0; e < epochs; e++) {
            unsigned int seed = 12345 + e * 7919 + omp_get_thread_num();
            mt19937 local_gen(seed);   // スレッドごとに独立したシード
            all_m[e] = noise_epoch(N, T, alpha, m0, sigma, local_gen);
        }

        // stepごとに mean, std を計算
        for (int t = 0; t <= T; t++) {
            vector<double> vals(epochs);
            for (int e = 0; e < epochs; e++) vals[e] = all_m[e][t];

            double mean = accumulate(vals.begin(), vals.end(), 0.0) / epochs;
            double sq_sum = 0.0;
            for (double v : vals) sq_sum += (v - mean) * (v - mean);
            double stddev = sqrt(sq_sum / epochs);

            csv << alpha << "," << t << "," << mean << "," << stddev << "\n";
        }
        cout << "Alpha " << alpha << " done.\n";
    }

    csv.close();
    cout << "結果を noise_Path_Hopfield.csv に保存しました。\n";
    return 0;
}