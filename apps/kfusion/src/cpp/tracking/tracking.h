 /*
 * SPDX-FileCopyrightText: 2017-2019 Smart Robotics Lab, Imperial College London
 * SPDX-FileCopyrightText: 2017-2019 Binbin Xu
 * SPDX-License-Identifier: BSD-3-Clause
 */

#ifndef OFUSION_TRACKING_H
#define OFUSION_TRACKING_H

#include "math_utils.h"
#include "commons.h"
#include "timings.h"
#include "../preprocessing/preprocessing.h"
#include <memory>
#include "../object/object.h"

class Tracking;
typedef std::shared_ptr<Tracking> TrackingPointer;

enum class Dataset{
  zr300,
  asus,
  icl_nuim,
  tum_rgbd
};

enum class RobustW{
  noweight,
  Huber,
  Tukey,
  Cauchy
};

class Tracking {
 public:
  Tracking();

  Tracking(const uint2 imgSize,
           const std::vector<int> &GN_opt_iterations,
           const bool use_rendered_depth,
           const bool using_ICP,
           const bool using_RGB);

  ~Tracking();

  void set_params_frame(const float4 k, const float* f_l_I, const float* f_l_D);
  void set_params_frame(const float4 k, const float* f_l_I,
                        const float*f_l_D, const cv::Mat& human_mask);

  bool trackLiveFrame(Matrix4& T_w_l,
                      const Matrix4& T_w_r,
                      const float4 k,
                      const float3* model_vertex,
                      const float3* model_normal);

  bool trackLiveFrame(Matrix4& T_w_l,
                      const Matrix4& T_w_r,
                      const float4 k,
                      const float3* model_vertex,
                      const float3* model_normal,
                      const cv::Mat& mask);

  void trackEachObject(ObjectPointer& objectptr,
                       const float4 k,
                       const Matrix4& T_w_c,
                       const Matrix4& T_w_r,
                       const float* f_l_I,
                       const float* f_l_D);

  bool checkPose(Matrix4 & pose, const Matrix4& oldPose);

  TrackData * getTrackingResult();

  void compute_residuals(const Matrix4& T_w_l,
                         const Matrix4& T_w_r,
                         const float3* w_refVertices_r,
                         const float3* w_refNormals_r,
                         const float3* w_refVertices_l,
                         const bool computeICP,
                         const bool computeRGB,
                         const float threshold);

// private:

  void obtainErrorParameters(const Dataset& dataset);

  void buildPyramid(const float* l_I, const float* l_D);

//  void buildPreviousPyramid(const float* r_I, const float* r_D);

  void calc_icp_cov(float3 *conv_icp,
                    const uint2 size,
                    const float *depth_image,
                    const Matrix4& K,
                    const float sigma_xy,
                    const float sigma_disp,
                    const float
                    baseline);

  void track_ICP_Kernel(TrackData *output,
                        const uint2 jacobian_size,
                        const float3 *inVertex,
                        const float3 *inNormal,
                        uint2 inSize,
                        const float3 *refVertex,
                        const float3 *refNormal,
                        uint2 refSize,
                        const Matrix4& Ttrack,
                        const Matrix4& view,
                        const float dist_threshold,
                        const float normal_threshold,
                        const float3 *icp_cov_layer);

  void track_ICP_Kernel(TrackData *output,
                        const uint2 jacobian_size,
                        const float3 *inVertex,
                        const float3 *inNormal,
                        uint2 inSize,
                        const float3 *refVertex,
                        const float3 *refNormal,
                        uint2 refSize,
                        const Matrix4& Ttrack,
                        const Matrix4& view,
                        const float dist_threshold,
                        const float normal_threshold,
                        const float3 *icp_cov_layer,
                        const cv::Mat& mask);
  /*
* left side notation:
* r_: reference coordinate
* w: world coordinate
* l: live coordinate
*/
  void trackRGB(TrackData *output,
                const uint2 jacobian_size,
                const float3 *r_vertices_render,
                const float3 *r_vertices_live,
                const float *r_image,
                const uint2 r_size,
                const float *l_image,
                const uint2 l_size,
                const float *l_gradx,
                const float *l_grady,
                const Matrix4& T_w_r,
                const Matrix4& T_w_l,
                const Matrix4& K,
                const float residual_criteria,
                const float grad_threshold,
                const float sigma_bright);

  void trackRGB(TrackData *output,
                const uint2 jacobian_size,
                const float3 *r_vertices_render,
                const float3 *r_vertices_live,
                const float *r_image,
                const uint2 r_size,
                const float *l_image,
                const uint2 l_size,
                const float *l_gradx,
                const float *l_grady,
                const Matrix4& T_w_r,
                const Matrix4& T_w_l,
                const Matrix4& K,
                const float residual_criteria,
                const float grad_threshold,
                const float sigma_bright,
                const cv::Mat& mask);

  void track_Obj_ICP_Kernel(TrackData *output,
                            const uint2 jacobian_size,
                            const float3 *c2_vertice_live,
                            const float3 *c2_normals_live,
                            uint2 inSize,
                            const float3 *c1_vertice_render,
                            const float3 *c1_Normals_render,
                            uint2 refSize,
                            const Matrix4& T_c2_o2, //to be estimated
                            const Matrix4& T_c1_o1,
                            const Matrix4& T_w_o1,
                            const float4 k,
                            const float dist_threshold,
                            const float normal_threshold,
                            const float3 *icp_cov_layer,
                            const cv::Mat& outlier_mask);

  void track_obj_RGB_kernel(TrackData* output, const uint2 jacobian_size,
                            const float3 *r_vertices_render,
                            const float3 *r_vertices_live, const float* r_image,
                            uint2 r_size, const float* l_image, uint2 l_size,
                            const float * l_gradx, const float * l_grady,
                            const Matrix4& T_c2_o2, //to be estimated
                            const Matrix4& T_c1_o1,
                            const Matrix4& K, const float residual_criteria,
                            const float grad_threshold, const float
                            sigma_bright,
                            const cv::Mat& outlier_mask);

  float2 obj_warp(float3& c1_vertex_o1, float3& o2_vertex,
                  const Matrix4& T_c1_o1, const Matrix4& T_c2_o2,
                  const Matrix4& K, const float3&c2_vertex);

  bool obj_icp_residual(float& residual, float3& o1_refNormal_o1,
                        float3& diff, const Matrix4& T_w_o1,
                        const float3 o2_vertex, const float3 *c1_vertice_render,
                        const float3 *c1_Normals_render,
                        const uint2& inSize, const float2& proj_pixel);
  /*!
 * warp warps a vertex from ref coordinate to live coordinate
 * @param l_vertex [in,out] returns the vertex in the live frame coordinate
 * @param w_vertex [in,out] returns the vertex in the world frame coordinate
 * @param T_w_r [in] transform points from reference frame coordinate to world coordinate
 * @param T_w_l [in] transfor points from live frame coordinate to world coordinate
 * @param K [in] intrinsic matrix
 * @param r_vertex [in] vertex in the reference frame coordinate
 * @return vertex in the reference frame
 */
  float2 warp(float3 &l_vertex,
              float3 &w_vertex,
              const Matrix4 &T_w_r,
              const Matrix4 &T_w_l,
              const Matrix4 &K,
              const float3 &r_vertex);

/*
 * Jacobian under the perturbation performed on T_w_l
 */
  bool rgb_jacobian(float J[6],
                    const float3 &l_vertex,
                    const float3 &w_vertex,
                    const Matrix4 &T_w_l,
                    const float2 &proj_pixel,
                    const float *l_gradx,
                    const float *l_grady,
                    const uint2 &l_size,
                    const Matrix4 &K,
                    const float grad_threshold,
                    const float weight);

  bool obj_rgb_jacobian(float J[6], const float3& l_vertex,
                        const float2& proj_pixel, const float* l_gradx,
                        const float* l_grady, const uint2& l_size,
                        const Matrix4& K, const float grad_threshold,
                        const float weight);

  float rgb_residual(const float *r_image,
                     const uint2 &r_pixel,
                     const uint2 &r_size,
                     const float *l_image,
                     const float2 &proj_pixel,
                     const uint2 &l_size);


  void reduceKernel(float *out, TrackData *J, const uint2 Jsize,
                    const uint2 size);

  void reduce(int blockIndex, float *out, TrackData *J, const uint2
  Jsize, const uint2 size, const int stack);

  bool solvePoseKernel(Matrix4 &pose_update, const float *output,
                       const float icp_threshold);

  bool checkPoseKernel(Matrix4 &pose, const Matrix4& oldPose,
                       const float *output, const uint2 imageSize,
                       const float track_threshold);

  //put current informatiopn for reference
  void setRefImgFromCurr();

  /*
 * backward rendering
 * warp the live image to the ref image position based on the T_w_l and T_w_r
 * and then calculate the residual image between warped image and ref image
 */

  void enable_ICP_tracker(const bool useICP){
    this->using_ICP_ = useICP;
  }

  void enable_RGB_tracker(const bool useRGB){
    this->using_RGB_ = useRGB;
  }

  void warp_to_residual(cv::Mat& warped_image, cv::Mat& residual,
                                  const float* l_image, const float* ref_image,
                                  const float3* r_vertex, const Matrix4& T_w_l,
                                  const Matrix4& T_w_r, const Matrix4& K,
                                  const uint2 outSize);

  void dump_residual(TrackData* res_data, const uint2 inSize);

  float calc_robust_residual(float residual,
                             const RobustW& robustfunction);

  float calc_robust_weight(float residual,
                           const RobustW& robustfunction);

  void compute_residual_kernel(TrackData* res_data,
                               const uint2 img_size,
                               const Matrix4& T_w_r,
                               const Matrix4& T_w_l,
                               const float4 k,
                               const float* r_image,
                               const float* l_image,
                               const float3* l_vertices,
                               const float3* l_normals,
                               const float3* w_refVertices_r,
                               const float3* w_refNormals_r,
                               const float3* l_refVertices_l,
                               const bool computeICP,
                               const bool computeRGB,
                               const float3 *icp_cov_layer,
                               const float sigma_bright,
                               const float threshold);

  void compute_icp_residual_kernel(TrackData& icp_row,
                                   const int in_index,
                                   const uint2 img_size,
                                   const Matrix4& T_w_l,
                                   const Matrix4& T_w_r,
                                   const Matrix4& K,
                                   const float3* l_vertices,
                                   const float3* l_normals,
                                   const float3* w_refVertices_r,
                                   const float3* w_refNormals_r,
                                   const float3 *icp_cov_layer,
                                   const float threshold);

  void compute_rgb_residual_kernel(TrackData& rgb_row,
                                   const uint2 in_pixel,
                                   const uint2 img_size,
                                   const Matrix4& T_w_l,
                                   const Matrix4& T_w_r,
                                   const Matrix4& K,
                                   const float* r_image,
                                   const float* l_image,
                                   const float3* l_refVertices_l,
                                   const float3* l_Vertices_l,
                                   const float sigma_bright,
                                   const float threshold);

 public:
  RobustW robustWeight_ = RobustW::Cauchy;
  static constexpr double huber_k = 1.345; //1.345
  static constexpr double tukey_b = 4.6851;  //4.6851
  static constexpr double cauchy_k = 2.3849; //2.3849

  const float mini_gradient_magintude_[3] = {0./255, 0./255, 0./255};
  static constexpr float rgb_residual_throsold = 8.0;
  const float rgb_tracking_threshold_[3] = {rgb_residual_throsold/4,
                                                   rgb_residual_throsold/2,
                                                   rgb_residual_throsold};

 private:
  float4 k_; //camera intrinsic matrix
  Matrix4 K_;
  uint2 imgSize_;

  std::vector<int> GN_opt_iterations_;
  bool use_live_depth_only_;
  bool using_ICP_;
  bool using_RGB_;

  float **reductionoutput_;
  TrackData **trackingResult_;

  float **l_D_;//depth pyramid on the live image (I_1)
  float * l_D_ref_;//depth pyramid on the live image (I_1)
  float **r_D_render_;//depth pyramid on the reference image (I_0)
  float **r_D_live_;//depth pyramid on the reference image (I_0)
  float **l_I_;//live image (I_1)
  float **r_I_;//reference image (I_0)

  float3 **l_vertex_;//Vertex seen from live frame
  float3 * l_vertex_ref_;//Vertex seen from live frame
  float3 **r_Vertex_live_;//vertex  seen from reference frame
  float3 **r_Vertex_render_;//vertex  seen from reference frame
  float3 **l_normal_;

  float **l_gradx_;
  float **l_grady_;
  float3 **icp_cov_pyramid_;

  float * gaussian;

  std::vector<uint2> localImgSize_;

  static constexpr float icp_threshold = 1e-05;
  static constexpr float occluded_depth_diff_ = 0.1f;
//  const float rgb_tracking_threshold_[3] = {2.0f, 2.0f, 2.0f};
//  const float mini_gradient_magintude[3] = {0.5/255., 0.3/255., 0.1/255.};

//  const float occluded_depth_diff = 0.1f;
  //icp&rgb covariance
  float sigma_bright_;
  float sigma_disparity_;
  float sigma_xy_;
  float baseline_;
  float focal_;

  int stack_;
  const bool use_virtual_camera_ = false;
  std::vector<cv::Mat> outlier_mask_;
  std::vector<cv::Mat> no_outlier_mask_;
  std::vector<cv::Mat> prev_outlier_mask_;

  static constexpr bool step_back_in_GN_ = true;
  const bool fill_in_missing_depth_rgb_residual_checking_ = false;

};

#endif //OFUSION_TRACKING_H
