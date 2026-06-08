// SPDX-License-Identifier: MIT
// Copyright (c) 2024

#pragma once

#include <Eigen/Core>
#include <Eigen/Eigenvalues>

#include <gtsam/linear/HessianFactor.h>
#include <gtsam_points/factors/integrated_vgicp_factor.hpp>

namespace gtsam_points {

/**
 * @brief Observability-aware VGICP factor for scan-to-map registration.
 *
 * The per-frame scan-to-map information matrix H (6x6) is eigen-decomposed:
 *   H = V diag(lambda) V^T
 *
 * Directions where lambda_i / lambda_max < tau are degenerate (corridor axis,
 * featureless ceiling, etc.). The factor shapes the linearized constraint so
 * that degenerate directions receive zero (hard) or attenuated (soft) information,
 * leaving them to IMU and scan-to-scan factors instead of an ill-conditioned
 * map gradient.
 *
 * Two shaping modes:
 *   hard (solution remapping, cf. Zhang et al., ICRA2016):
 *     s_i = 1 if lambda_i / lambda_max >= tau, else 0
 *     H' = V diag(s * lambda) V^T,  b' = V diag(s) V^T b
 *
 *   soft (Tikhonov-style continuous attenuation):
 *     s_i = lambda_i / (lambda_i + reg * lambda_max)
 *     H' = V diag(s * lambda) V^T,  b' = V diag(s) V^T b
 *
 * Note: H' = V diag(s*lambda) V^T (not P H P^T) so that gain s_i is applied
 * linearly to each eigenvalue, not squared.
 */
template <typename SourceFrame = gtsam_points::PointCloud>
class IntegratedObservabilityAwareVGICPFactor_ : public IntegratedVGICPFactor_<SourceFrame> {
public:
  using Base = IntegratedVGICPFactor_<SourceFrame>;
  using shared_ptr = gtsam_points::shared_ptr<IntegratedObservabilityAwareVGICPFactor_>;

  IntegratedObservabilityAwareVGICPFactor_(
    const gtsam::Pose3& fixed_target_pose,
    gtsam::Key source_key,
    const GaussianVoxelMap::ConstPtr& target_voxels,
    const std::shared_ptr<const SourceFrame>& source)
  : Base(fixed_target_pose, source_key, target_voxels, source) {}

  ~IntegratedObservabilityAwareVGICPFactor_() override = default;

  void set_degeneracy_threshold(double tau) { degeneracy_threshold_ = tau; }
  void set_regularization(double reg) { lambda_reg_ = reg; }
  void set_soft(bool soft) { soft_ = soft; }

  double min_eigenvalue_ratio() const { return min_eig_ratio_; }
  const Eigen::Matrix<double, 6, 1>& observability_gains() const { return gains_; }

  gtsam::GaussianFactor::shared_ptr linearize(const gtsam::Values& values) const override {
    const auto base = Base::linearize(values);
    if (!base) return base;

    const auto hf = std::dynamic_pointer_cast<gtsam::HessianFactor>(base);
    if (!hf || hf->size() != 1) return base;

    // augmentedInformation() returns the (n+1)x(n+1) augmented matrix [H, b; b^T, f].
    // Extract the top-left 6x6 information block explicitly.
    const Eigen::MatrixXd aug = hf->augmentedInformation();
    const int n = aug.rows() - 1;  // == 6 for Pose3
    const Eigen::MatrixXd H = aug.topLeftCorner(n, n);
    const Eigen::VectorXd b = hf->linearTerm();  // n-vector
    const double f = hf->constantTerm();

    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(H);
    if (es.info() != Eigen::Success) return base;

    const Eigen::VectorXd& evals = es.eigenvalues();   // ascending
    const Eigen::MatrixXd& V = es.eigenvectors();
    const double lambda_max = evals.maxCoeff();

    min_eig_ratio_ = (lambda_max > 0.0) ? std::max(0.0, evals.minCoeff() / lambda_max) : 0.0;

    // Compute per-direction gains s_i in [0, 1]
    Eigen::VectorXd s(n);
    for (int i = 0; i < n; i++) {
      if (lambda_max <= 0.0) {
        s[i] = 0.0;
      } else if (soft_) {
        s[i] = evals[i] / (evals[i] + lambda_reg_ * lambda_max + 1e-12);
      } else {
        s[i] = (evals[i] / lambda_max >= degeneracy_threshold_) ? 1.0 : 0.0;
      }
    }
    gains_ = s;

    // Shape: H' = V diag(s * lambda) V^T,  b' = V diag(s) V^T b
    // This applies s_i linearly to each eigenvalue (not squared as in P H P^T).
    const Eigen::MatrixXd Hp = V * (s.cwiseProduct(evals)).asDiagonal() * V.transpose();
    const Eigen::VectorXd bp = V * s.asDiagonal() * V.transpose() * b;

    return gtsam::HessianFactor::shared_ptr(
      new gtsam::HessianFactor(this->keys()[0], Hp, bp, f));
  }

  gtsam::NonlinearFactor::shared_ptr clone() const override {
    return gtsam::NonlinearFactor::shared_ptr(new IntegratedObservabilityAwareVGICPFactor_(*this));
  }

private:
  double degeneracy_threshold_ = 0.02;
  double lambda_reg_ = 0.05;
  bool soft_ = false;

  mutable double min_eig_ratio_ = 1.0;
  mutable Eigen::Matrix<double, 6, 1> gains_ = Eigen::Matrix<double, 6, 1>::Ones();
};

using IntegratedObservabilityAwareVGICPFactor = IntegratedObservabilityAwareVGICPFactor_<>;

}  // namespace gtsam_points
