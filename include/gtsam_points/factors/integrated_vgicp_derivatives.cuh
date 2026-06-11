// SPDX-License-Identifier: MIT
// Copyright (c) 2021  Kenji Koide (k.koide@aist.go.jp)

#pragma once

#include <memory>
#include <Eigen/Core>
#include <Eigen/Geometry>

#include <gtsam_points/types/gaussian_voxelmap_gpu.hpp>

struct CUstream_st;

namespace gtsam_points {

class LinearizedSystem6;
class TempBufferManager;

class IntegratedVGICPDerivatives {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  IntegratedVGICPDerivatives(
    const GaussianVoxelMapGPU::ConstPtr& target,
    const PointCloud::ConstPtr& source,
    CUstream_st* ext_stream,
    std::shared_ptr<TempBufferManager> temp_buffer);
  ~IntegratedVGICPDerivatives();

  void set_inlier_update_thresh(double trans, double angle) {
    inlier_update_thresh_trans = trans;
    inlier_update_thresh_angle = angle;
  }

  void set_enable_offloading(bool enable) { enable_offloading = enable; }

  void set_enable_surface_validation(bool enable) { enable_surface_validation = enable; }

  // Per-correspondence GNC (Geman-McClure). noise_bound is the fixed outlier scale c^2 floor;
  // when adaptive, c^2 is estimated per linearization from the residual median (only loosens
  // above the floor). mu is annealed from mu_init toward 1 across linearizations.
  void set_gnc(bool enable, bool adaptive, double noise_bound, double chi2_quantile, double mu_init, double mu_decay) {
    gnc_enabled = enable;
    gnc_adaptive = adaptive;
    gnc_c2 = noise_bound;
    gnc_chi2_quantile = chi2_quantile;
    gnc_mu_init = mu_init;
    gnc_mu_decay = mu_decay;
    gnc_mu = -1.0;  // (re)initialized to mu_init on the next linearize
  }

  int get_num_inliers() const { return num_inliers; }

  void touch_points();

  // synchronized interface
  LinearizedSystem6 linearize(const Eigen::Isometry3f& x);
  double compute_error(const Eigen::Isometry3f& xl, const Eigen::Isometry3f& xe);

  void reset_inliers(const Eigen::Isometry3f& x, const Eigen::Isometry3f* d_x, bool force_update = false);
  void update_inliers(int num_inliers);

  // async interface
  void sync_stream();
  void issue_linearize(const Eigen::Isometry3f* d_x, LinearizedSystem6* d_output);
  void issue_compute_error(const Eigen::Isometry3f* d_xl, const Eigen::Isometry3f* d_xe, float* d_output);

private:
  bool enable_offloading;

  bool enable_surface_validation;
  double inlier_update_thresh_trans;
  double inlier_update_thresh_angle;

  bool external_stream;
  CUstream_st* stream;
  std::shared_ptr<TempBufferManager> temp_buffer;

  GaussianVoxelMapGPU::ConstPtr target;
  PointCloud::ConstPtr source;

  Eigen::Isometry3f inlier_evaluation_point;
  const Eigen::Isometry3f* inlier_evaluation_point_gpu;

  int num_inliers;
  int* num_inliers_gpu;
  int* source_inliers;

  // Per-correspondence GNC state (Geman-McClure; weighting applied in vgicp_derivatives_kernel).
  bool gnc_enabled = false;
  bool gnc_adaptive = true;
  double gnc_c2 = 1.0;               // nominal floor noise bound c^2
  double gnc_chi2_quantile = 7.815;  // chi-square(3) quantile for the adaptive bound
  double gnc_mu_init = 100.0;
  double gnc_mu_decay = 0.7;
  double gnc_mu = -1.0;  // current control parameter; <0 = uninitialized
};
}  // namespace gtsam_points