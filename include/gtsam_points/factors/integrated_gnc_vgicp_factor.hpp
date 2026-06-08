// SPDX-License-Identifier: MIT
// Copyright (c) 2024

#pragma once

#include <cmath>

#include <gtsam/linear/HessianFactor.h>
#include <gtsam_points/factors/integrated_vgicp_factor.hpp>

namespace gtsam_points {

/**
 * @brief GNC (Graduated Non-Convexity) robust VGICP factor for scan-to-map registration.
 *
 * Uses the Geman-McClure robust cost with a graduated annealing schedule:
 *
 *   rho(r2; mu, c2) = mu * c2 * r2 / (mu * c2 + r2)
 *   weight w = (mu * c2 / (mu * c2 + r2))^2
 *
 * where c2 = noise_bound^2 and mu is annealed (mu *= mu_decay each linearization).
 * As mu → ∞ the cost approaches L2; as mu → 0 strongly deweights large residuals.
 *
 * Reference: Yang et al., "Graduated Non-Convexity for Robust Spatial Perception",
 * ICRA 2020.
 */
template <typename SourceFrame = gtsam_points::PointCloud>
class IntegratedGNCVGICPFactor_ : public IntegratedVGICPFactor_<SourceFrame> {
public:
  using Base = IntegratedVGICPFactor_<SourceFrame>;
  using shared_ptr = gtsam_points::shared_ptr<IntegratedGNCVGICPFactor_>;

  IntegratedGNCVGICPFactor_(
    const gtsam::Pose3& fixed_target_pose,
    gtsam::Key source_key,
    const GaussianVoxelMap::ConstPtr& target_voxels,
    const std::shared_ptr<const SourceFrame>& source)
  : Base(fixed_target_pose, source_key, target_voxels, source) {}

  ~IntegratedGNCVGICPFactor_() override = default;

  void set_noise_bound(double c) { gnc_c2_ = c * c; }
  void set_mu_init(double mu) { mu_ = mu; }
  void set_mu_decay(double decay) { gnc_mu_decay_ = decay; }

  double current_mu() const { return mu_; }
  double last_weight() const { return last_weight_; }

  gtsam::GaussianFactor::shared_ptr linearize(const gtsam::Values& values) const override {
    const auto base = Base::linearize(values);
    if (!base) return base;

    const auto hf = std::dynamic_pointer_cast<gtsam::HessianFactor>(base);
    if (!hf || hf->size() != 1) return base;

    // augmentedInformation() returns the (n+1)x(n+1) augmented [H, b; b^T, f].
    // Extract the top-left n×n block.
    const Eigen::MatrixXd aug = hf->augmentedInformation();
    const int n = aug.rows() - 1;  // == 6 for Pose3
    const Eigen::MatrixXd H = aug.topLeftCorner(n, n);
    const Eigen::VectorXd b = hf->linearTerm();
    const double f = hf->constantTerm();

    // Residual squared from the constant term of the Hessian factor
    const double r2 = std::max(0.0, f);

    // Geman-McClure weight
    const double denom = mu_ * gnc_c2_ + r2;
    const double w = (denom > 0.0) ? std::pow(mu_ * gnc_c2_ / denom, 2.0) : 0.0;
    last_weight_ = w;

    // Anneal mu toward zero each linearization
    mu_ *= gnc_mu_decay_;

    const Eigen::MatrixXd Hw = w * H;
    const Eigen::VectorXd bw = w * b;

    return gtsam::HessianFactor::shared_ptr(
      new gtsam::HessianFactor(this->keys()[0], Hw, bw, w * f));
  }

  gtsam::NonlinearFactor::shared_ptr clone() const override {
    return gtsam::NonlinearFactor::shared_ptr(new IntegratedGNCVGICPFactor_(*this));
  }

private:
  double gnc_c2_ = 1.0;        // noise_bound^2
  mutable double mu_ = 100.0;  // annealing param (decreases each linearize)
  double gnc_mu_decay_ = 0.7;

  mutable double last_weight_ = 1.0;
};

using IntegratedGNCVGICPFactor = IntegratedGNCVGICPFactor_<>;

}  // namespace gtsam_points
