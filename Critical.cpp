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

// 初期状態を作る（m0 を反映）
vector<double> x_initial(int N, const VectorXd &xsi, double m0, mt19937 &gen) {
    vector<int> indices(N);
    iota(indices.begin(), indices.end(), 0);
    shuffle(indices.begin(), indices.end(), gen);

    VectorXd xsi_new = xsi;
    int flip_count = static_cast<int>((1 - m0) / 2 * N);
    for (int i = 0; i < flip_count; i++) {
        int idx = indices[i];
        xsi_new(idx) *= -1;
    }
    vector<double> result(N);
    for (int i = 0; i < N; i++) result[i] = xsi_new(i);
    return result;
}

// Path Integral Hopfield の 1 試行
vector<double> calc(int N, double Alpha, int T, double m0, mt19937 &gen) {
    MatrixXd x = MatrixXd::Zero(T+1, N);
    MatrixXd phi = MatrixXd::Zero(T+1, N);
    MatrixXd Q = MatrixXd::Zero(T+1, T+1);
    MatrixXd R_hat = MatrixXd::Zero(T+1, T+1);
    MatrixXd S = MatrixXd::Zero(T+1, T+1);
    MatrixXd S_hat = MatrixXd::Zero(T+1, T+1);

    normal_distribution<double> normal(0.0, 1.0);
    MatrixXd Z(T+1, N);
    for (int i = 0; i < T+1; i++)
        for (int j = 0; j < N; j++)
            Z(i, j) = normal(gen);

    VectorXd xsi = VectorXd::Random(N).unaryExpr([](double v) { return v >= 0 ? 1.0 : -1.0; });
    vector<double> x0 = x_initial(N, xsi, m0, gen);
    for (int i = 0; i < N; i++) x(0, i) = x0[i];

    for (int i = 0; i <= T; i++) Q(i, i) = 1;
    R_hat(0, 0) = Alpha;
    double epsilon = 1e-3;

    VectorXd h(N);

    for (int t = 0; t < T; t++) {
        SelfAdjointEigenSolver<MatrixXd> es(R_hat.topLeftCorner(t+1, t+1));
        VectorXd D = es.eigenvalues().cwiseMax(0).array() + epsilon;
        MatrixXd U = es.eigenvectors();
        MatrixXd R_k = U * D.asDiagonal() * U.transpose();
        MatrixXd C = R_k.llt().matrixL();

        phi.row(t) = (C.row(t) * Z.topRows(t+1)).transpose();

        double xsi_x_ave = xsi.dot(x.row(t)) / N;

        #pragma omp parallel for
        for (int n = 0; n < N; n++) {
            double term = xsi_x_ave * xsi(n) + phi(t, n)
                        - (S_hat.block(0, t, t+1, 1).transpose() * x.block(0, n, t+1, 1))(0);
            if (t > 0) term -= Alpha * x(t, n);
            h(n) = term;
        }

        #pragma omp parallel for
        for (int n = 0; n < N; n++)
            x(t+1, n) = (h(n) >= 0 ? 1.0 : -1.0);

        VectorXd x_next = x.row(t+1);

        #pragma omp parallel for
        for (int i = 0; i <= t; i++) {
            double dot = x_next.dot(x.row(i)) / N;
            Q(t+1, i) = Q(i, t+1) = dot;
        }

        if (t == 0) {
            S(1, 0) = -2.0 / sqrt(2 * M_PI * Alpha) * exp(-m0 * m0 / (2 * Alpha));
        } else {
            VectorXd x_phi_ave(t+1);
            for (int i = 0; i <= t; i++) {
                x_phi_ave(i) = x_next.dot(phi.row(i)) / N;
            }
            S.block(t+1, 0, 1, t+1) =
                -x_phi_ave.transpose() *
                (R_hat.topLeftCorner(t+1, t+1) + epsilon * MatrixXd::Identity(t+1, t+1)).inverse();
        }

        MatrixXd I = MatrixXd::Identity(t+2, t+2);
        R_hat.topLeftCorner(t+2, t+2) =
            Alpha * (I + S.topLeftCorner(t+2, t+2)).inverse() *
            Q.topLeftCorner(t+2, t+2) *
            (I + S.topLeftCorner(t+2, t+2).transpose()).inverse();

        S_hat.topLeftCorner(t+2, t+2) =
            -Alpha * (I + S.topLeftCorner(t+2, t+2)).inverse().transpose();
    }

    vector<double> m(T+1);
    for (int i = 0; i <= T; i++)
        m[i] = xsi.dot(x.row(i)) / N;

    return m;
}

// ---- α に対する m_final の平均 ----
double mean_m_final(int N, int T, double m0, double alpha, int epochs, mt19937 &gen) {
    double sum = 0.0;

    #pragma omp parallel
    {
        // 各スレッドごとに異なる乱数生成器を作る
        mt19937 local_gen(gen() + omp_get_thread_num() * 7919);
        double local_sum = 0.0;

        #pragma omp for nowait
        for (int e = 0; e < epochs; e++) {
            vector<double> m = calc(N, alpha, T, m0, local_gen);
            local_sum += m.back();
        }

        // 部分和を合計に反映
        #pragma omp atomic
        sum += local_sum;
    }

    return sum / epochs;
}

// ---- 二分探索で臨界点を探す ----
double find_alpha_threshold(int N, int T, double m0,
                            mt19937 &gen, double alpha_min, double alpha_max,
                            double tol, double threshold, int epochs) {
    while (alpha_max - alpha_min > tol) {
        double alpha_mid = 0.5 * (alpha_min + alpha_max);

        double m_final_mean = mean_m_final(N, T, m0, alpha_mid, epochs, gen);

        if (m_final_mean >= threshold) {
            alpha_min = alpha_mid; // まだ安定
        } else {
            alpha_max = alpha_mid; // 崩壊
        }
    }

    return 0.5 * (alpha_min + alpha_max);
}

int main() {
    int N = 100000;
    int T = 50;
    double m0 = 0.8;
    double threshold = 0.98;
    double tol = 1e-3;
    int epochs = 10;

    mt19937 gen(1234);

    double alpha_c = find_alpha_threshold(N, T, m0, gen, 0.05, 0.3, tol, threshold, epochs);

    cout << "alpha_c = " << alpha_c << endl;

    return 0;
}