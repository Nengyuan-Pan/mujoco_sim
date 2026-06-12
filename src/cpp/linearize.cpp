// Analytical dynamics linearization (batch, along trajectory)
// For each (X[k], U[k]): compute A_k(12x12), B_k(12x6), x_next_k(12)

#include "types.h"

namespace {

// Solve 6x6 linear system M * X = I, output M_inv
// Uses Gaussian elimination with partial pivoting (no external deps)
void invert_6x6(const mjtNum* M, double* M_inv) {
    double A[6][12] = {};
    for (int i = 0; i < 6; ++i) {
        for (int j = 0; j < 6; ++j)
            A[i][j] = M[i * 6 + j];
        A[i][6 + i] = 1.0;
    }
    for (int col = 0; col < 6; ++col) {
        int pivot_row = col;
        double max_val = (A[col][col] < 0 ? -A[col][col] : A[col][col]);
        for (int row = col + 1; row < 6; ++row) {
            double a = (A[row][col] < 0 ? -A[row][col] : A[row][col]);
            if (a > max_val) { max_val = a; pivot_row = row; }
        }
        if (max_val < 1e-12) continue;
        if (pivot_row != col) {
            double tmp[12];
            std::memcpy(tmp, A[col], sizeof(tmp));
            std::memcpy(A[col], A[pivot_row], sizeof(tmp));
            std::memcpy(A[pivot_row], tmp, sizeof(tmp));
        }
        double inv = 1.0 / A[col][col];
        for (int j = 0; j < 12; ++j) A[col][j] *= inv;
        for (int row = 0; row < 6; ++row) {
            if (row == col) continue;
            double factor = A[row][col];
            for (int j = 0; j < 12; ++j)
                A[row][j] -= factor * A[col][j];
        }
    }
    for (int i = 0; i < 6; ++i)
        for (int j = 0; j < 6; ++j)
            M_inv[i * 6 + j] = A[i][6 + j];
}

// Single-point analytical linearization
// A_out: (12,12), B_out: (12,6), x_next_out: (12,)
//
// actuator_mode: 0=torque (tau=u), 1=position (tau=Kp*(u-q)-Kd*qdot)
// kp, kd: (6,) gain arrays, nullptr when actuator_mode==0
void linearize_analytical_single(
    mjModel* m, mjData* d,
    const double* x, const double* u,
    const double* init_q_left,
    double eps, double dt,
    int actuator_mode,
    const double* kp, const double* kd,
    double* A_out, double* B_out, double* x_next_out)
{
    set_arm_forward(m, d, x, x + kNQ, init_q_left);

    // ---- 1. Mass matrix M (6x6) ----
    mjtNum M_full[60*60];
    std::memset(M_full, 0, sizeof(M_full));
    mj_fullM(m, M_full, d->qM);
    mjtNum M[6*6];
    for (int i = 0; i < 6; ++i)
        std::memcpy(M + i * 6, M_full + i * m->nv, 6 * sizeof(mjtNum));
    double M_inv[6*6];
    invert_6x6(M, M_inv);

    // ---- 2. Bias force h(q, qdot) ----
    std::memset(d->qacc, 0, m->nv * sizeof(mjtNum));
    mjtNum h_base[30];
    mj_rne(m, d, 0, h_base);

    // ---- 3. dh/dq (central differences, 6 perturbations) ----
    double H_q[6][6] = {};
    double q_save[6];
    std::memcpy(q_save, d->qpos, 6 * sizeof(double));
    for (int j = 0; j < 6; ++j) {
        std::memcpy(d->qpos, q_save, 6 * sizeof(double));
        d->qpos[j] += eps;
        mj_forward(m, d);
        std::memset(d->qacc, 0, m->nv * sizeof(mjtNum));
        mjtNum h_plus[30];
        mj_rne(m, d, 0, h_plus);

        std::memcpy(d->qpos, q_save, 6 * sizeof(double));
        d->qpos[j] -= eps;
        mj_forward(m, d);
        std::memset(d->qacc, 0, m->nv * sizeof(mjtNum));
        mjtNum h_minus[30];
        mj_rne(m, d, 0, h_minus);

        for (int i = 0; i < 6; ++i)
            H_q[i][j] = (h_plus[i] - h_minus[i]) / (2.0 * eps);
    }
    std::memcpy(d->qpos, q_save, 6 * sizeof(double));

    // ---- 4. dh/dqdot (central differences, 6 perturbations) ----
    double H_qdot[6][6] = {};
    double qd_save[6];
    std::memcpy(qd_save, d->qvel, 6 * sizeof(double));
    for (int j = 0; j < 6; ++j) {
        std::memcpy(d->qvel, qd_save, 6 * sizeof(double));
        d->qvel[j] += eps;
        mj_forward(m, d);
        std::memset(d->qacc, 0, m->nv * sizeof(mjtNum));
        mjtNum h_plus[30];
        mj_rne(m, d, 0, h_plus);

        std::memcpy(d->qvel, qd_save, 6 * sizeof(double));
        d->qvel[j] -= eps;
        mj_forward(m, d);
        std::memset(d->qacc, 0, m->nv * sizeof(mjtNum));
        mjtNum h_minus[30];
        mj_rne(m, d, 0, h_minus);

        for (int i = 0; i < 6; ++i)
            H_qdot[i][j] = (h_plus[i] - h_minus[i]) / (2.0 * eps);
    }
    std::memcpy(d->qvel, qd_save, 6 * sizeof(double));
    mj_forward(m, d);

    // ---- 5. Assemble A_c, B_c (mode-dependent) ----
    //
    // Torque mode (actuator_mode == 0):
    //   tau = ctrl
    //   A_c = [0, I; -M^{-1}*H_q, -M^{-1}*H_qdot]
    //   B_c = [0; M^{-1}]
    //
    // Position mode (actuator_mode == 1):
    //   tau = Kp*(ctrl - q) - Kd*qdot
    //   A_c = [0, I; -M^{-1}*(H_q+diag(Kp)), -M^{-1}*(H_qdot+diag(Kd))]
    //   B_c = [0; M^{-1}*diag(Kp)]
    double A_c[12][12] = {};
    for (int i = 0; i < 6; ++i) A_c[i][6 + i] = 1.0;

    double M_inv_Hq[6][6] = {};
    double M_inv_Hqdot[6][6] = {};
    for (int i = 0; i < 6; ++i) {
        for (int j = 0; j < 6; ++j) {
            double sq = 0.0, sqd = 0.0;
            for (int k = 0; k < 6; ++k) {
                sq  += M_inv[i*6 + k] * H_q[k][j];
                sqd += M_inv[i*6 + k] * H_qdot[k][j];
            }
            M_inv_Hq[i][j]    = -sq;
            M_inv_Hqdot[i][j] = -sqd;
        }
    }
    for (int i = 0; i < 6; ++i) {
        for (int j = 0; j < 6; ++j) {
            A_c[6 + i][j]      = M_inv_Hq[i][j];
            A_c[6 + i][6 + j]  = M_inv_Hqdot[i][j];
        }
    }

    double B_c[12][6] = {};
    if (actuator_mode == 0) {
        // Torque mode: B_c = [0; M^{-1}]
        for (int i = 0; i < 6; ++i)
            for (int j = 0; j < 6; ++j)
                B_c[6 + i][j] = M_inv[i*6 + j];
    } else {
        // Safety: kp/kd must be non-null when actuator_mode == 1
        // (enforced by Python-side configure_actuator_mode())
        assert(kp && kd);
        // Position mode:
        //   B_c[6+i][j] = M_inv[i,j] * Kp[j]   (= M^{-1} * diag(Kp))
        //   A_c[6+i][j]   += -M_inv[i,j] * Kp[j] (= -M^{-1} * diag(Kp))
        //   A_c[6+i][6+j] += -M_inv[i,j] * Kd[j] (= -M^{-1} * diag(Kd))
        for (int i = 0; i < 6; ++i) {
            for (int j = 0; j < 6; ++j) {
                double mi = M_inv[i*6 + j];
                B_c[6 + i][j]      = mi * kp[j];
                A_c[6 + i][j]     += -mi * kp[j];
                A_c[6 + i][6 + j] += -mi * kd[j];
            }
        }
    }

    // ---- 6. Euler discretization ----
    for (int i = 0; i < 12; ++i) {
        for (int j = 0; j < 12; ++j)
            A_out[i * 12 + j] = (i == j ? 1.0 : 0.0) + A_c[i][j] * dt;
    }
    for (int i = 0; i < 12; ++i)
        for (int j = 0; j < 6; ++j)
            B_out[i * 6 + j] = B_c[i][j] * dt;

    // ---- 7. Baseline next state ----
    // Note: u[] is assumed pre-clipped to actuator_ctrlrange by the Python caller,
    // so the x_next computed by mj_step (which clips internally) matches the dynamics.
    set_arm_forward(m, d, x, x + kNQ, init_q_left);
    for (int i = 0; i < kNU; ++i)
        d->ctrl[i] = clip(u[i], -1e10, 1e10);
    // Zero left arm ctrl (no PD needed for next-state prediction in linearization)
    std::memset(d->ctrl + kNU, 0, kNU * sizeof(mjtNum));
    mj_step(m, d);
    std::memcpy(x_next_out, d->qpos, kNQ * sizeof(double));
    std::memcpy(x_next_out + kNQ, d->qvel, kNQ * sizeof(double));
}

}  // anonymous namespace
