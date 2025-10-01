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

// x_initial: ランダムに (1-m0)/2 割合のスピンを反転
vector<double> x_initial(int N, const VectorXd &xsi, double m0, mt19937 &gen) {
    vector<int> indices(N);
    iota(indices.begin(), indices.end(), 0);
    shuffle(indices.begin(), indices.end(), gen);

    VectorXd xsi_new = xsi;
    int flip_count = static_cast<int>((1 - m0) / 2 * N);
    for (int i = 0; i < flip_count; i++) {
        xsi_new(indices[i]) *= -1;
    }
    vector<double> result(N);
    for (int i = 0; i < N; i++) result[i] = xsi_new(i);
    return result;
}

// Python版 calc(N, Alpha, T, m0, sigma) と同じロジック
vector<double> calc(int N, double Alpha, int T, double m0, double sigma, mt19937 &gen) {
    MatrixXd x   = MatrixXd::Zero(T+1, N);
    MatrixXd phi = MatrixXd::Zero(T+1, N);
    MatrixXd Q   = MatrixXd::Identity(T+1, T+1);
    MatrixXd R_hat = MatrixXd::Zero(T+1, T+1);
    MatrixXd S     = MatrixXd::Zero(T+1, T+1);
    MatrixXd S_hat = MatrixXd::Zero(T+1, T+1);

    MatrixXd Z    = MatrixXd::Zero(T+1, N);   // ~ N(0,1)
    MatrixXd SK_Z = MatrixXd::Zero(T+1, N);   // ~ N(0,1)
    MatrixXd SK_Phi = MatrixXd::Zero(T+1, N); // SK 由来ノイズ

    normal_distribution<double> nd(0.0, 1.0);
    for (int i = 0; i <= T; i++) {
        for (int j = 0; j < N; j++) {
            Z(i, j)    = nd(gen);
            SK_Z(i, j) = nd(gen);
        }
    }

    // 初期条件
    VectorXd xsi = VectorXd::Ones(N);
    vector<double> x0 = x_initial(N, xsi, m0, gen);
    for (int i = 0; i < N; i++) x(0, i) = x0[i];

    R_hat(0, 0) = Alpha;
    const double epsilon = 1e-4;

    VectorXd h(N);

    // 時間発展
    for (int t = 0; t < T; t++) {
        // ---- phi[t] の生成:  R_hat[0:t,0:t] の正定値化 → コレスキー → C[t]*Z[0:t]
        {
            MatrixXd Rh = R_hat.topLeftCorner(t+1, t+1);
            SelfAdjointEigenSolver<MatrixXd> es(Rh);
            VectorXd D = es.eigenvalues().cwiseMax(0.0).array() + epsilon;
            MatrixXd U = es.eigenvectors();
            MatrixXd Rk = U * D.asDiagonal() * U.transpose();
            MatrixXd C = Rk.llt().matrixL(); // Cholesky: L

            // phi[t] = C[t] @ Z[0:t]
            phi.row(t) = (C.row(t) * Z.topRows(t+1)).transpose();
        }

        // ---- SK_Phi[t] の生成:  eig(sigma*Q[0:t,0:t]) → |D| + eps → chol → L[t]*SK_Z[0:t]
        {
            MatrixXd QQ = (sigma * Q.topLeftCorner(t+1, t+1)).eval();
            SelfAdjointEigenSolver<MatrixXd> es(QQ);
            VectorXd D = es.eigenvalues().cwiseAbs(); // |D|
            MatrixXd U = es.eigenvectors();
            MatrixXd regulared = U * (D.asDiagonal().toDenseMatrix() + epsilon * MatrixXd::Identity(t+1, t+1)) * U.transpose();
            MatrixXd L = regulared.llt().matrixL();
            SK_Phi.row(t) = (L.row(t) * SK_Z.topRows(t+1)).transpose();
        }

        // ---- h の更新
        double xsi_x_ave = xsi.dot(x.row(t)) / N;

        // h(n) を並列で更新
        #pragma omp parallel for
        for (int n = 0; n < N; n++) {
            //  - S_hat[0:t,t]^T @ x[0:t,n]
            double term_Shat = (S_hat.block(0, t, t+1, 1).transpose()
                               * x.block(0, n, t+1, 1))(0, 0);

            //  - sigma * S[t,0:t] @ x[0:t,n]
            double term_Ssigma = 0.0;
            if (t >= 0) {
                term_Ssigma = (S.block(t, 0, 1, t+1)
                              * x.block(0, n, t+1, 1))(0, 0);
            }

            double term = xsi_x_ave * xsi(n)
                        + phi(t, n)
                        + SK_Phi(t, n)
                        - term_Shat
                        - sigma * term_Ssigma;

            if (t > 0) term -= Alpha * x(t, n);

            h(n) = term;
        }

        // x の符号更新（並列）
        #pragma omp parallel for
        for (int n = 0; n < N; n++) {
            x(t+1, n) = (h(n) >= 0 ? 1.0 : -1.0);
        }

        // Q の更新（並列）
        VectorXd x_next = x.row(t+1);
        #pragma omp parallel for
        for (int i = 0; i <= t; i++) {
            double dot = x_next.dot(x.row(i)) / N;
            Q(t+1, i) = Q(i, t+1) = dot;
        }

        // S の更新
        if (t == 0) {
            // Python: S[1,0] = - 2 / sqrt(2π(Alpha+sigma)) * exp(-m0^2/(2(Alpha+sigma)))
            S(1, 0) = -2.0 / sqrt(2.0 * M_PI * (Alpha + sigma))
                      * exp(- (m0 * m0) / (2.0 * (Alpha + sigma)));
        } else {
            VectorXd x_phi_ave(t+1);
            for (int i = 0; i <= t; i++) {
                x_phi_ave(i) = x_next.dot(phi.row(i)) / N;
            }
            MatrixXd invR = (R_hat.topLeftCorner(t+1, t+1)
                           + epsilon * MatrixXd::Identity(t+1, t+1)).inverse();
            S.block(t+1, 0, 1, t+1) = - x_phi_ave.transpose() * invR;
        }

        // R_hat, S_hat の更新
        MatrixXd I = MatrixXd::Identity(t+2, t+2);
        MatrixXd Stl = S.topLeftCorner(t+2, t+2);
        MatrixXd Qtl = Q.topLeftCorner(t+2, t+2);
        MatrixXd iL = (I + Stl).inverse();
        MatrixXd iR = (I + Stl.transpose()).inverse();

        R_hat.topLeftCorner(t+2, t+2) = Alpha * iL * Qtl * iR;
        S_hat.topLeftCorner(t+2, t+2) = - Alpha * iL.transpose();
    }

    // m(t) = x(t)・xsi / N  (xsi=1 なので平均スピン)
    vector<double> m(T+1);
    for (int i = 0; i <= T; i++)
        m[i] = xsi.dot(x.row(i)) / N;

    return m;
}

// ---- m_final を epochs 回平均（エポック並列化）----
double mean_m_final(int N, int T, double m0, double alpha, double sigma,
                    int epochs, mt19937 &gen) {
    double sum = 0.0;

    #pragma omp parallel for reduction(+:sum)
    for (int e = 0; e < epochs; e++) {
        // 各スレッドで独立した乱数生成器を作る
        mt19937 local_gen(gen() + e + omp_get_thread_num() * 1000);

        vector<double> m = calc(N, alpha, T, m0, sigma, local_gen);
        sum += m.back();

        
    }

    return sum / epochs;
}

// ---- 二分探索で臨界点を探す (平均化) ----
double find_alpha_threshold(int N, int T, double m0, double sigma,
                            mt19937 &gen, double alpha_min, double alpha_max,
                            double tol, double threshold, int epochs) {
    while (alpha_max - alpha_min > tol) {
        double alpha_mid = 0.5 * (alpha_min + alpha_max);

        double m_final_mean = mean_m_final(N, T, m0, alpha_mid, sigma, epochs, gen);

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
    double threshold = 0.8;
    double tol = 1e-3;
    int epochs = 10;   // 平均回数

    vector<double> Sigmas ;
    for (double i = 0.0; i <= 0.41; i += 0.01) Sigmas.push_back(i);

    ofstream fout("noise_Critical.csv");
    fout << "sigma,alpha_c\n";

    #pragma omp parallel for
    for (int idx = 0; idx < Sigmas.size(); idx++) {
        double sigma = Sigmas[idx];
        mt19937 gen(1234 + idx * 100 + omp_get_thread_num());

        double alpha_c = find_alpha_threshold(N, T, m0, sigma, gen, 0.0001, 0.3, tol, threshold, epochs);

        #pragma omp critical
        {
            fout << sigma << "," << alpha_c << "\n";
            cout << "sigma=" << sigma << " -> alpha_c=" << alpha_c << endl;
        }
    }

    fout.close();
    return 0;
}