// Forward pass: single step and linesearch variants

#include "types.h"

namespace fwd {

/// Single forward pass (MPC mode, fixed alpha=0.5)
/// Returns true if trajectory is valid (all finite)
bool single(
    py::array_t<double> X_new_a, py::array_t<double> U_new_a,
    py::array_t<double> X_nom_a, py::array_t<double> U_nom_a,
    py::array_t<double> Ks_a,   py::array_t<double> ks_a,
    uintptr_t model_ptr, uintptr_t data_ptr,
    const double* init_q_left,
    const double* ctrl_lo, const double* ctrl_hi,
    double alpha)
{
    mjModel* m = to_model(model_ptr);
    mjData* d = to_data(data_ptr);

    int N = static_cast<int>(U_nom_a.shape(0));

    double* X_new = X_new_a.mutable_data();
    double* U_new = U_new_a.mutable_data();
    const double* X_nom = X_nom_a.data();
    const double* U_nom = U_nom_a.data();
    const double* Ks   = Ks_a.data();
    const double* ks   = ks_a.data();

    std::memcpy(X_new, X_nom, kNX * sizeof(double));

    for (int k = 0; k < N; ++k) {
        double dx[12];
        for (int i = 0; i < kNX; ++i)
            dx[i] = X_new[k * kNX + i] - X_nom[k * kNX + i];

        const double* K_k = Ks + k * (kNX * kNU);
        const double* k_k = ks + k * kNU;
        for (int i = 0; i < kNU; ++i) {
            double sum = 0.0;
            for (int j = 0; j < kNX; ++j)
                sum += K_k[i * kNX + j] * dx[j];
            U_new[k * kNU + i] = U_nom[k * kNU + i] + alpha * k_k[i] + sum;
            U_new[k * kNU + i] = clip(U_new[k * kNU + i], ctrl_lo[i], ctrl_hi[i]);
        }

        sim_step(m, d,
                 X_new + k * kNX, X_new + k * kNX + kNQ,
                 U_new + k * kNU,
                 init_q_left, ctrl_lo, ctrl_hi,
                 X_new + (k + 1) * kNX, X_new + (k + 1) * kNX + kNQ);

        for (int i = 0; i < kNX; ++i) {
            if (!std::isfinite(X_new[(k + 1) * kNX + i]))
                return false;
        }
    }
    return true;
}

/// Returns: py::tuple of (accepted: bool, X_out, U_out, cost_out: float)
py::tuple linesearch(
    py::array_t<double> X_nom_a, py::array_t<double> U_nom_a,
    py::array_t<double> Ks_a,   py::array_t<double> ks_a,
    py::array_t<double> alpha_list_a,
    double cost_old,
    uintptr_t model_ptr, uintptr_t data_ptr,
    const double* init_q_left,
    const double* ctrl_lo, const double* ctrl_hi,
    py::object cost_fn)
{
    int N = static_cast<int>(U_nom_a.shape(0));
    int n_alpha = static_cast<int>(alpha_list_a.size());

    // Pre-allocate temp buffers
    std::vector<py::ssize_t> shape_X = {static_cast<py::ssize_t>(N + 1), kNX};
    std::vector<py::ssize_t> shape_U = {static_cast<py::ssize_t>(N), kNU};
    py::array_t<double> X_tmp(shape_X);
    py::array_t<double> U_tmp(shape_U);

    const double* alpha_list = alpha_list_a.data();

    py::array_t<double> X_best = X_nom_a;
    py::array_t<double> U_best = U_nom_a;
    double cost_best = cost_old;

    for (int ia = 0; ia < n_alpha; ++ia) {
        double alpha = alpha_list[ia];
        bool ok = single(X_tmp, U_tmp, X_nom_a, U_nom_a,
                         Ks_a, ks_a,
                         model_ptr, data_ptr,
                         init_q_left, ctrl_lo, ctrl_hi, alpha);
        if (!ok) continue;

        double cost_new;
        try {
            cost_new = cost_fn(X_tmp, U_tmp).cast<double>();
        } catch (py::error_already_set&) {
            PyErr_Clear();
            continue;
        }

        if (cost_new < cost_best) {
            X_best = X_tmp;
            U_best = U_tmp;
            cost_best = cost_new;
            return py::make_tuple(true, X_best, U_best, cost_best);
        }
    }
    return py::make_tuple(false, X_best, U_best, cost_best);
}

}  // namespace fwd
