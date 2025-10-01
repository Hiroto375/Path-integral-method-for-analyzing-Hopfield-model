#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <Eigen/Dense>
#include <omp.h>
using namespace std;
using namespace Eigen;

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

int main() {
    int N = 100000;   // ニューロン数
    int T = 50;       // ステップ数

    // alpha と m0 のリスト
    vector<double> alphas;
    for (double a = 0.01; a <= 0.20; a += 0.01) alphas.push_back(a);

    vector<double> m0s;
    for (double m0 = 0.1; m0 <= 1.0; m0 += 0.1) m0s.push_back(m0);

    ofstream fout("Heatmap.csv");
    fout << "alpha,m0,m_final\n";

    #pragma omp parallel for collapse(2) schedule(dynamic)
    for (int i = 0; i < (int)m0s.size(); i++) {
        for (int j = 0; j < (int)alphas.size(); j++) {
            double m0 = m0s[i];
            double alpha = alphas[j];

            mt19937 gen(1234 + 7919*(i*alphas.size()+j) + omp_get_thread_num());

            auto m = calc(N, alpha, T, m0, gen);
            double m_final = m[T];   // T ステップ後の重なり

            #pragma omp critical
            fout << alpha << "," << m0 << "," << m_final << "\n";
            cout << "(" << alpha << ", " << m0 << ")" << " done.\n";
        }
    }

    fout.close();
    cout << "結果を Heatmap.csv に保存しました。\n";
    return 0;
}