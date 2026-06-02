// Common types and utility functions for C++ accelerated iLQR
// No Chinese comments to avoid MSVC encoding issues

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <mujoco/mujoco.h>
#include <algorithm>
#include <cstring>
#include <cmath>
#include <vector>

namespace py = pybind11;

constexpr int kNQ = 6;
constexpr int kNX = 12;
constexpr int kNU = 6;

// Cast uintptr_t from Python to MuJoCo struct pointers
inline mjModel* to_model(uintptr_t ptr) { return reinterpret_cast<mjModel*>(ptr); }
inline mjData* to_data(uintptr_t ptr) { return reinterpret_cast<mjData*>(ptr); }

// Clamp x to [lo, hi]
inline double clip(double x, double lo, double hi) {
    return x < lo ? lo : (x > hi ? hi : x);
}

// Set arm state (right arm q/qdot, left arm fixed) and run mj_forward
inline void set_arm_forward(mjModel* m, mjData* d,
                             const double* q, const double* qdot,
                             const double* init_q_left) {
    std::memcpy(d->qpos, q, kNQ * sizeof(double));
    std::memcpy(d->qvel, qdot, kNQ * sizeof(double));
    std::memcpy(d->qpos + kNQ, init_q_left, kNQ * sizeof(double));
    std::memset(d->qvel + kNQ, 0, kNQ * sizeof(double));
    mj_forward(m, d);
}

// Single simulation step with control clipping and left arm PD
inline void sim_step(mjModel* m, mjData* d,
                     const double* q, const double* qdot, const double* u,
                     const double* init_q_left,
                     const double* ctrl_lo, const double* ctrl_hi,
                     double* q_out, double* qdot_out) {
    std::memcpy(d->qpos, q, kNQ * sizeof(double));
    std::memcpy(d->qvel, qdot, kNQ * sizeof(double));
    std::memcpy(d->qpos + kNQ, init_q_left, kNQ * sizeof(double));
    std::memset(d->qvel + kNQ, 0, kNQ * sizeof(double));
    // Right arm control with clipping
    for (int i = 0; i < kNU; ++i)
        d->ctrl[i] = clip(u[i], ctrl_lo[i], ctrl_hi[i]);
    // Left arm PD hold (match Python env.step behavior)
    for (int i = 0; i < kNU; ++i) {
        double err_q = init_q_left[i] - d->qpos[kNQ + i];
        double err_qd = -d->qvel[kNQ + i];
        d->ctrl[kNQ + i] = clip(200.0 * err_q - 20.0 * err_qd, ctrl_lo[i], ctrl_hi[i]);
    }
    mj_step(m, d);
    std::memcpy(q_out, d->qpos, kNQ * sizeof(double));
    std::memcpy(qdot_out, d->qvel, kNQ * sizeof(double));
}
