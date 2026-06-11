// SPDX-License-Identifier: MIT
// Copyright (c) 2021  Kenji Koide (k.koide@aist.go.jp)

#include <gtsam_points/factors/integrated_vgicp_derivatives.cuh>

#include <iostream>
#include <thrust/remove.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/system/cuda/execution_policy.h>

#include <cuda_runtime.h>
#include <cub/device/device_reduce.cuh>

#include <gtsam_points/cuda/kernels/pose.cuh>
#include <gtsam_points/cuda/kernels/untie.cuh>
#include <gtsam_points/cuda/kernels/lookup_voxels.cuh>
#include <gtsam_points/cuda/kernels/linearized_system.cuh>
#include <gtsam_points/cuda/kernels/vgicp_derivatives.cuh>
#include <gtsam_points/cuda/stream_temp_buffer_roundrobin.hpp>

#include <gtsam_points/types/gaussian_voxelmap_gpu.hpp>

namespace gtsam_points {

void IntegratedVGICPDerivatives::issue_linearize(const Eigen::Isometry3f* d_x, LinearizedSystem6* d_output) {
  //
  lookup_voxels_kernel corr_kernel(enable_surface_validation, *target, source->points_gpu, source->normals_gpu, d_x);
  auto corr_first = thrust::make_transform_iterator(source_inliers, corr_kernel);

  // GNC: anneal mu toward 1 across linearizations and pass mu*c^2 to the per-point weight.
  float gnc_mu_c2 = 1.0f;
  if (gnc_enabled) {
    if (gnc_mu < 0.0) gnc_mu = gnc_mu_init;  // first linearization of this factor
    double c2_used = gnc_c2;                 // nominal floor

    if (gnc_adaptive && num_inliers > 0) {
      // Adaptive noise bound from this frame's residual scale (matches the CPU path):
      // e_i ~ chi-square(3) for inliers, so s^2 = median(e_i)/median(chi2_3) and
      // c^2 = s^2 * chi2_quantile. The adaptive bound may only LOOSEN above the floor.
      vgicp_error_kernel err_kernel(d_x, d_x, *target, source->points_gpu, source->covs_gpu);
      auto err_first = thrust::make_transform_iterator(corr_first, err_kernel);
      thrust::device_vector<float> d_errs(num_inliers);
      thrust::copy_n(thrust::cuda::par.on(stream), err_first, num_inliers, d_errs.begin());
      thrust::sort(thrust::cuda::par.on(stream), d_errs.begin(), d_errs.end());
      cudaStreamSynchronize(stream);
      const float median = d_errs[num_inliers / 2];
      constexpr double chi2_3_median = 2.3660;  // median of chi-square with 3 dof
      const double c2 = (static_cast<double>(median) / chi2_3_median) * gnc_chi2_quantile;
      if (c2 > c2_used) c2_used = c2;
    }

    gnc_mu_c2 = static_cast<float>(gnc_mu * c2_used);
    const double next = gnc_mu * gnc_mu_decay;
    gnc_mu = (next < 1.0) ? 1.0 : next;
  }

  vgicp_derivatives_kernel deriv_kernel(d_x, *target, source->points_gpu, source->covs_gpu, gnc_enabled, gnc_mu_c2);
  auto first = thrust::make_transform_iterator(corr_first, deriv_kernel);

  void* temp_storage = nullptr;
  size_t temp_storage_bytes = 0;

  cub::DeviceReduce::Reduce(
    temp_storage,
    temp_storage_bytes,
    first,
    d_output,
    num_inliers,
    thrust::plus<LinearizedSystem6>(),
    LinearizedSystem6::zero(),
    stream);

  temp_storage = temp_buffer->get_buffer(temp_storage_bytes);

  cub::DeviceReduce::Reduce(
    temp_storage,
    temp_storage_bytes,
    first,
    d_output,
    num_inliers,
    thrust::plus<LinearizedSystem6>(),
    LinearizedSystem6::zero(),
    stream);
}

}  // namespace gtsam_points