 /*
 * SPDX-FileCopyrightText: 2017-2019 Smart Robotics Lab, Imperial College London
 * SPDX-FileCopyrightText: 2017-2019 Binbin Xu
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "tracking.h"



Tracking::Tracking(const uint2 imgSize,
                   const std::vector<int> &GN_opt_iterations,
                   const bool use_live_depth_only,
                   const bool using_ICP,
                   const bool using_RGB)
    : imgSize_(make_uint2(imgSize.x, imgSize.y)),
      GN_opt_iterations_(GN_opt_iterations),
      use_live_depth_only_(use_live_depth_only),
      using_ICP_(using_ICP),
      using_RGB_(using_RGB)
{
  reductionoutput_ = (float **) calloc(sizeof(float *) * GN_opt_iterations.size(), 1);
  trackingResult_ =
      (TrackData **) calloc(sizeof(TrackData *) * GN_opt_iterations.size(), 1);

  l_D_ = (float **) calloc(sizeof(float *) * GN_opt_iterations.size(), 1);
  l_D_ref_ = (float *) calloc(sizeof(float) * (imgSize.x * imgSize.y), 1);
  r_D_live_ = (float **) calloc(sizeof(float *) * GN_opt_iterations.size(), 1);
  r_D_render_ = (float **) calloc(sizeof(float *) * GN_opt_iterations.size(), 1);
  l_I_ = (float **) calloc(sizeof(float *) * GN_opt_iterations.size(), 1);
  r_I_ = (float **) calloc(sizeof(float *) * GN_opt_iterations.size(), 1);
  l_vertex_ = (float3 **) calloc(sizeof(float3 *) * GN_opt_iterations.size(), 1);
  l_vertex_ref_ = (float3 *) calloc(sizeof(float3) * (imgSize.x * imgSize.y), 1);
  r_Vertex_live_ = (float3 **) calloc(sizeof(float3 *) * GN_opt_iterations.size(), 1);
  r_Vertex_render_ = (float3 **) calloc(sizeof(float3 *) * GN_opt_iterations.size(), 1);
  l_normal_ = (float3 **) calloc(sizeof(float3 *) * GN_opt_iterations.size(), 1);
  l_gradx_ = (float **) calloc( sizeof(float *) * GN_opt_iterations.size(), 1);
  l_grady_ = (float **) calloc( sizeof(float *) * GN_opt_iterations.size(), 1);

  icp_cov_pyramid_ =
      (float3 **) calloc(sizeof(float3 *) * GN_opt_iterations.size(), 1);

  for (int level = 0; level < GN_opt_iterations.size(); ++level) {
    uint2 localimagesize = make_uint2(
        imgSize.x / (int) pow(2, level),
        imgSize.y / (int) pow(2, level));
    localImgSize_.push_back(localimagesize); //from fine to coarse
    reductionoutput_[level] = (float *) calloc(sizeof(float) * 8 * 32, 1);
    trackingResult_[level] = (TrackData *) calloc(
        2 * sizeof(TrackData) * imgSize.x * imgSize.y, 1);
    l_D_[level] = (float *) calloc(sizeof(float) * (imgSize.x * imgSize.y)
                                       / (int) pow(2, 2 * level), 1);
    r_D_live_[level] = (float *) calloc(sizeof(float) * (imgSize.x * imgSize.y)
                                            / (int) pow(2, level), 1);
    r_D_render_[level] = (float *) calloc(sizeof(float) * (imgSize.x * imgSize.y)
                                              / (int) pow(2, level), 1);
    l_I_[level] = (float *) calloc(sizeof(float) * (imgSize.x * imgSize.y)
                                       / (int) pow(2, 2 * level), 1);
    r_I_[level] = (float *) calloc(sizeof(float) * (imgSize.x * imgSize.y)
                                       / (int) pow(2, 2 * level), 1);
    l_vertex_[level] = (float3 *) calloc(sizeof(float3) * (imgSize.x * imgSize.y)
                                             / (int) pow(2, 2 * level), 1);
    r_Vertex_live_[level] = (float3 *) calloc(sizeof(float3) * (imgSize.x * imgSize.y)
                                             / (int) pow(2, level), 1);
    r_Vertex_render_[level] = (float3 *) calloc(sizeof(float3) * (imgSize.x * imgSize.y)
                                             / (int) pow(2, level), 1);
    l_normal_[level] = (float3 *) calloc(sizeof(float3) * (imgSize.x * imgSize.y)
                                             / (int) pow(2, 2 * level), 1);
    l_gradx_[level] = (float *) calloc(sizeof(float) * (imgSize.x * imgSize.y)
                                           / (int) pow(2, 2 * level), 1);
    l_grady_[level] = (float *) calloc(sizeof(float) * (imgSize.x * imgSize.y)
                                           / (int) pow(2, 2 * level), 1);
    icp_cov_pyramid_[level] = (float3 *) calloc(sizeof(float3) * (imgSize.x * imgSize.y)
                                                    / (int) pow(2, 2 * level), 1);
  }

  // ********* BEGIN : Generate the gaussian *************
  size_t gaussianS = radius * 2 + 1;
  gaussian = (float*) calloc(gaussianS * sizeof(float), 1);
  int x;
  for (unsigned int i = 0; i < gaussianS; i++) {
    x = i - 2;
    gaussian[i] = expf(-(x * x) / (2 * delta * delta));
  }
  // ********* END : Generate the gaussian *************

  if (using_ICP ^ using_RGB) stack_ = 1;
  if (using_ICP && using_RGB) stack_ = 2;

//  if (this->robustWeight == RobustW::noweight){
//    float mini_gradient_magintude[3] = {0.2 / 255., 0.2 / 255., 0.2 / 255.};
//    float rgb_tracking_threshold[3] = {0.15, 0.15, 0.15};
//    memcpy(this->mini_gradient_magintude_, mini_gradient_magintude,
//           sizeof(float) * 3);
//    memcpy(this->rgb_tracking_threshold_, rgb_tracking_threshold,
//           sizeof(float) * 3);
//
//  }
  for (int level = 0; level < GN_opt_iterations_.size(); ++level) {
    cv::Mat outlier_mask = cv::Mat::zeros(localImgSize_[level].y,
                                          localImgSize_[level].x, CV_8UC1);
    no_outlier_mask_.push_back(outlier_mask);
  }
}

Tracking::~Tracking() {
  for (unsigned int i = 0; i < GN_opt_iterations_.size(); ++i) {
    free(l_D_[i]);
    free(r_D_live_[i]);
    free(r_D_render_[i]);
    free(l_I_[i]);
    free(r_I_[i]);
    free(l_vertex_[i]);
    free(r_Vertex_live_[i]);
    free(r_Vertex_render_[i]);
    free(l_normal_[i]);
    free(l_gradx_[i]);
    free(l_grady_[i]);
    free(icp_cov_pyramid_[i]);
    free(trackingResult_[i]);
    free(reductionoutput_[i]);
  }
  free(l_D_);
  free(l_D_ref_);
  free(r_D_live_);
  free(r_D_render_);
  free(l_I_);
  free(r_I_);
  free(l_vertex_);
  free(l_vertex_ref_);
  free(r_Vertex_live_);
  free(r_Vertex_render_);
  free(l_normal_);
  free(l_gradx_);
  free(l_grady_);
  free(icp_cov_pyramid_);
  free(gaussian);
  free(trackingResult_);
  free(reductionoutput_);
}

void Tracking::set_params_frame(const float4 k,
                                const float* f_l_I,
                                const float* f_l_D){

  k_ = k;
  K_ = getCameraMatrix(k);

  //  const float weight_rgb = 0.1f;  unused. use R^-1 instead
//  const float weight_icp = 1.0f;  unused. use R^-1 instead

  // pre-compute for ICP tracking
  if (k.y < 0){//ICL-NUIM
    obtainErrorParameters(Dataset::icl_nuim);
  }
  else{//TUM
    obtainErrorParameters(Dataset::tum_rgbd);
  }

  buildPyramid(f_l_I, f_l_D);

  for (int level = GN_opt_iterations_.size() - 1; level >= 0; --level) {
    if (using_ICP_){
      calc_icp_cov(icp_cov_pyramid_[level], localImgSize_[level], l_D_[level],
                   getCameraMatrix(k_ / (1 << level)), sigma_xy_, sigma_disparity_,
                   baseline_);
    }
  }

   outlier_mask_ = no_outlier_mask_;
}


void Tracking::set_params_frame(const float4 k, const float* f_l_I,
                                const float*f_l_D, const cv::Mat& human_mask){
  k_ = k;
  K_ = getCameraMatrix(k);

  //  const float weight_rgb = 0.1f;  unused. use R^-1 instead
//  const float weight_icp = 1.0f;  unused. use R^-1 instead

  // pre-compute for ICP tracking
  if (k.y < 0){//ICL-NUIM
    obtainErrorParameters(Dataset::icl_nuim);
  }
  else{//TUM
    obtainErrorParameters(Dataset::tum_rgbd);
  }

  buildPyramid(f_l_I, f_l_D);

  for (int level = GN_opt_iterations_.size() - 1; level >= 0; --level) {
    if (using_ICP_){
      calc_icp_cov(icp_cov_pyramid_[level], localImgSize_[level], l_D_[level],
                   getCameraMatrix(k_ / (1 << level)), sigma_xy_, sigma_disparity_,
                   baseline_);
    }
  }

  for (int level = 0; level < GN_opt_iterations_.size(); ++level) {
    cv::Mat outlier_mask;
    cv::Size mask_size = human_mask.size()/((1 << level));
    cv::resize(human_mask, outlier_mask, mask_size);
    outlier_mask_.push_back(outlier_mask);
  }
}

bool Tracking::trackLiveFrame(Matrix4& T_w_l,
                              const Matrix4& T_w_r,
                              const float4 k,
                              const float3* model_vertex,
                              const float3* model_normal)
{

//  T_w_l = T_w_r;  //initilize the new live pose

//  //rendering the reference depth information from model
  if (using_RGB_ && (!use_live_depth_only_)){
    for (unsigned int i = 0; i < GN_opt_iterations_.size(); ++i) {
      Matrix4 invK = getInverseCameraMatrix(k_ / float(1 << i));
      if (i == 0){
          vertex2depth(r_D_render_[0], model_vertex, imgSize_, T_w_r);
          depth2vertexKernel(r_Vertex_render_[0], r_D_render_[0],
                             localImgSize_[0], invK);

        //memcpy(r_Vertex_[0], model_vertex,
        //       sizeof(float3) * localImgSize_[0].x * localImgSize_[0].y);
      }
      else{
        //using the rendered depth
        halfSampleRobustImageKernel(r_D_render_[i], r_D_render_[i - 1],
                                    localImgSize_[i-1], e_delta * 3, 1);
        depth2vertexKernel(r_Vertex_render_[i], r_D_render_[i],
                           localImgSize_[i], invK);

      }
    }
  }



  Matrix4 pose_update;

  Matrix4 previous_pose = T_w_l;

  //coarse-to-fine iteration
  for (int level = GN_opt_iterations_.size() - 1; level >= 0; --level) {
    float previous_error = INFINITY;
    for (int i = 0; i < GN_opt_iterations_[level]; ++i) {
      if (using_ICP_){
        const Matrix4 projectReference = getCameraMatrix(k_) * inverse(T_w_r);
        track_ICP_Kernel(trackingResult_[level], imgSize_, l_vertex_[level],
                         l_normal_[level], localImgSize_[level], model_vertex,
                         model_normal, imgSize_, T_w_l, projectReference,
                         dist_threshold, normal_threshold,
                         icp_cov_pyramid_[level], outlier_mask_[level]);
      }

      if (using_RGB_){
        //render reference image to live image -- opposite to the original function call
        if (use_live_depth_only_){
          trackRGB(trackingResult_[level] + imgSize_.x * imgSize_.y,
                   imgSize_, r_Vertex_live_[level],r_Vertex_live_[level],
                   r_I_[level], localImgSize_[level],
                   l_I_[level], localImgSize_[level], l_gradx_[level],
                   l_grady_[level], T_w_r, T_w_l, getCameraMatrix(k_ / (1 << level)),
                   rgb_tracking_threshold_[level],
                   mini_gradient_magintude_[level], sigma_bright_,
                   prev_outlier_mask_[level]);
        }
        else{
          trackRGB(trackingResult_[level] + imgSize_.x * imgSize_.y,
                   imgSize_, r_Vertex_render_[level],r_Vertex_live_[level],
                   r_I_[level], localImgSize_[level],
                   l_I_[level], localImgSize_[level], l_gradx_[level],
                   l_grady_[level], T_w_r, T_w_l, getCameraMatrix(k_ / (1 << level)),
                   rgb_tracking_threshold_[level],
                   mini_gradient_magintude_[level], sigma_bright_,
                   prev_outlier_mask_[level]);
        }
      }

      if (using_ICP_ && (!using_RGB_)){
        reduceKernel(reductionoutput_[level], trackingResult_[level],
                     imgSize_, localImgSize_[level]);
      }

      if ((!using_ICP_) && using_RGB_){
        reduceKernel(reductionoutput_[level],
                     trackingResult_[level]+ imgSize_.x * imgSize_.y,
                     imgSize_, localImgSize_[level]);
      }

      if (using_ICP_ && using_RGB_){
        reduceKernel(reductionoutput_[level], trackingResult_[level],
                     imgSize_, localImgSize_[level]);
      }


      const float current_error =
          reductionoutput_[level][0]/reductionoutput_[level][28];
//      std::cout<< "Level " << level << ", Iteration " << i << ", Error: " << current_error<<std::endl;

      if (current_error > (previous_error /* + 1e-1f*/)){
//        std::cout<< "Level " << level << " Iteration " << i << " - Cost increase from " << previous_error << " to " <<
//                 current_error<<std::endl;
        if (step_back_in_GN_){
          T_w_l = previous_pose;
        }

        /*const Matrix4 projectReference = getCameraMatrix(k_) * inverse(T_w_r);
        track_ICP_Kernel(trackingResult_[level], imgSize_, l_vertex_[level],
                         l_normal_[level], localImgSize_[level], model_vertex,
                         model_normal, imgSize_, T_w_l, projectReference,
                         dist_threshold, normal_threshold, icp_cov_pyramid_[level]);
        reduceKernel(reductionoutput_[level], trackingResult_[level],
                     imgSize_, localImgSize_[level]);
        step_back_error = reductionoutput_[level][0]/reductionoutput_[level][28];
//        std::cout<< "Level " << level << " step back " << ", Error: " <<
//                 step_back_error<<std::endl;
        assert(step_back_error == previous_error);*/
        break;
      }
      previous_error = current_error;

      if (solvePoseKernel(pose_update, reductionoutput_[level],
                          icp_threshold)) {
//        previous_pose = T_w_l;
//        T_w_l = pose_update * previous_pose;
        break;
      }

      previous_pose = T_w_l;
      T_w_l = pose_update * previous_pose;
//      printMatrix4("updated live pose", T_w_l);
    }
  }

  //check the pose issue
//  bool tracked = checkPoseKernel(T_w_l, T_w_r, reductionoutput_, imgSize_,
//                                 track_threshold);
    bool tracked = true;


/*  const Matrix4 projectReference = getCameraMatrix(k_) * inverse(T_w_r);
  track_ICP_Kernel(trackingResult_[0], imgSize_, l_vertex_[0],
                   l_normal_[0], localImgSize_[0], model_vertex,
                   model_normal, imgSize_, T_w_l, projectReference,
                   dist_threshold, normal_threshold, icp_cov_pyramid_[0]);
  reduceKernel(reductionoutput_[0], trackingResult_[0],
               imgSize_, localImgSize_[0]);
  final_error = reductionoutput_[0][0]/reductionoutput_[0][28];
//  std::cout<< "Final Level " << ", Error: " << final_error<<std::endl;
  assert(step_back_error == final_error);*/
  return tracked;

}

float2 Tracking::obj_warp(float3& c1_vertex_o1, float3& o2_vertex,
                          const Matrix4& T_c1_o1, const Matrix4& T_c2_o2,
                          const Matrix4& K, const float3& c2_vertex) {
  o2_vertex = inverse(T_c2_o2) * c2_vertex;
  c1_vertex_o1 = T_c1_o1 * o2_vertex;
  const float3 proj_vertex = rotate(K, c1_vertex_o1);
  return make_float2(proj_vertex.x/proj_vertex.z,
                     proj_vertex.y/proj_vertex.z);
}


bool Tracking::obj_icp_residual(float& residual, float3& o1_refNormal_o1,
                                float3& diff, const Matrix4& T_w_o1,
                                const float3 o2_vertex, const float3 *c1_vertice_render,
                                const float3 *c1_Normals_render,
                                const uint2& inSize, const float2& proj_pixel)
{
  float3 w_refVertex_o1 = bilinear_interp(c1_vertice_render, inSize,
      proj_pixel);
  float3 w_refNormal_o1 = bilinear_interp(c1_Normals_render, inSize,
      proj_pixel);
//  const uint2 refPixel = make_uint2(proj_pixel.x, proj_pixel.y);
//  float3 w_refVertex_o1 =c1_vertice_render[refPixel.x + refPixel.y * inSize.x];
//  float3 w_refNormal_o1 =c1_Normals_render[refPixel.x + refPixel.y * inSize.x];

  if (w_refNormal_o1.x == INVALID) return false;
  float3 o1_refVertex_o1 = inverse(T_w_o1) * w_refVertex_o1;
  o1_refNormal_o1 = rotate(inverse(T_w_o1), w_refNormal_o1);
  diff =  o2_vertex - o1_refVertex_o1;
  residual = dot(o1_refNormal_o1, diff);
  return true;
}

void Tracking::track_Obj_ICP_Kernel(TrackData *output,
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
                                    const cv::Mat& outlier_mask) {
  TICK();
  uint2 pixel = make_uint2(0, 0);
  unsigned int pixely, pixelx;
#pragma omp parallel for \
      shared(output), private(pixel, pixelx, pixely)
  for (pixely = 0; pixely < inSize.y; pixely++) {
    for (pixelx = 0; pixelx < inSize.x; pixelx++) {

      pixel.x = pixelx;
      pixel.y = pixely;
      const unsigned idx = pixel.x + pixel.y * jacobian_size.x;
      TrackData &row = output[idx];

      if (outlier_mask.at<uchar>(pixely, pixelx) != 0) {
        row.result = -6;
        continue;
      }
      if (c2_normals_live[pixel.x + pixel.y * inSize.x].x == INVALID) {
        row.result = -1;
        continue;
      }

      const float3 c2_vertex = c2_vertice_live[pixel.x + pixel.y * inSize.x];
      if ((c2_vertex.x == 0) && (c2_vertex.y == 0) && (c2_vertex.z == 0)) {
        row.result = -1;
        continue;
      }

      float3 o2_vertex, c1_vertex_o1;
      const float2 projpixel = obj_warp(c1_vertex_o1, o2_vertex, T_c1_o1,
          T_c2_o2, getCameraMatrix(k), c2_vertex);

      if (projpixel.x < 1 || projpixel.x > refSize.x - 2 ||
          projpixel.y < 1 || projpixel.y > refSize.y - 2 ||
          std::isnan(projpixel.x) || std::isnan(projpixel.y)) {
        row.result = -2;
        continue;
      }

      const float3 c_normal = c2_normals_live[pixel.x + pixel.y * inSize.x];
      const float3 o2_normal = rotate(inverse(T_c2_o2), c_normal);

      float residual;
      float3 o1_refNormal_o1, diff;
      bool has_residual = obj_icp_residual(residual, o1_refNormal_o1, diff,
          T_w_o1, o2_vertex, c1_vertice_render, c1_Normals_render, refSize, projpixel);

      if (!has_residual) {
        row.result = -3;
        continue;
      }

      if (length(diff) > dist_threshold) {
        row.result = -4;
        continue;
      }
      if (dot(o2_normal, o1_refNormal_o1) < normal_threshold) {
        row.result = -5;
        continue;
      }

      //calculate the inverse of covariance as weights
      const float3 P = icp_cov_layer[pixel.x + pixel.y * inSize.x];
      const float sigma_icp = o1_refNormal_o1.x * o1_refNormal_o1.x * P.x
          + o1_refNormal_o1.y * o1_refNormal_o1.y * P.y
          + o1_refNormal_o1.z * o1_refNormal_o1.z * P.z;
      const float inv_cov = sqrtf(1.0 / sigma_icp);

      row.error = inv_cov * residual;

//      float3 Jtrans = rotate(o1_refNormal_o1, transpose(T_c2_o2));
      float3 Jtrans = rotate(T_c2_o2, o1_refNormal_o1);
      ((float3 *) row.J)[0] = -1.0f * inv_cov * - 1.0f * Jtrans;
      ((float3 *) row.J)[1] = /*-1.0f */ inv_cov * cross(c2_vertex, Jtrans);

      row.result = 1;
    }
  }
}




void Tracking::track_obj_RGB_kernel(TrackData* output, const uint2 jacobian_size,
                          const float3 *r_vertices_render,
                          const float3 *r_vertices_live, const float* r_image,
                          uint2 r_size, const float* l_image, uint2 l_size,
                          const float * l_gradx, const float * l_grady,
                                    const Matrix4& T_c2_o2, //to be estimated
                                    const Matrix4& T_c1_o1,
                          const Matrix4& K, const float residual_criteria,
                          const float grad_threshold, const float sigma_bright,
                                    const cv::Mat& outlier_mask) {
  TICK();
  uint2 r_pixel = make_uint2(0, 0);
  unsigned int r_pixely, r_pixelx;
#pragma omp parallel for shared(output),private(r_pixel,r_pixelx,r_pixely)
  for (r_pixely = 0; r_pixely < r_size.y; r_pixely++) {
    for (r_pixelx = 0; r_pixelx < r_size.x; r_pixelx++) {
      r_pixel.x = r_pixelx;
      r_pixel.y = r_pixely;

      TrackData & row = output[r_pixel.x + r_pixel.y * jacobian_size.x];

      if (outlier_mask.at<uchar>(r_pixely, r_pixelx) != 0) {
        row.result = -6;
        continue;
      }

      const int r_index = r_pixel.x + r_pixel.y * r_size.x;
      float3 r_vertex_render = r_vertices_render[r_index];
      const float3 r_vertex_live = r_vertices_live[r_index];

      //if rendered depth is not available
      if ((r_vertex_render.z <= 0.f) || (r_vertex_render.z == INVALID)) {
        //if live depth is not availvle too =>depth error
//        if (r_vertex_live.z <= 0.f ||r_vertex_live.z == INVALID) {
        row.result = -1;
        continue;
//        }

        /*else{
//          if live depth is availvle, use live depth instead
//          would introduce occlusion however
          r_vertex_render = r_vertex_live;
        }*/
      }

      //if the difference between rendered and live depth is too large =>occlude
      if (length(r_vertex_render - r_vertex_live) > occluded_depth_diff_){
        //not in the case that no live depth
        if (r_vertex_live.z > 0.f){
          row.result = -3;
          continue;
        }
      }


      float3 o1_vertex, c2_vertex_o1;
      const float2 projpixel = obj_warp(c2_vertex_o1, o1_vertex,
          T_c2_o2, T_c1_o1, K, r_vertex_render);

      if (projpixel.x < 1  || projpixel.x > l_size.x - 2 || projpixel.y < 1
          || projpixel.y > l_size.y - 2
          ) {
        row.result = -2;
        continue;
      }

      const float residual = rgb_residual(r_image, r_pixel, r_size, l_image, projpixel, l_size);
      const float inv_cov = 1.0/sigma_bright;
      bool gradValid = obj_rgb_jacobian(row.J, c2_vertex_o1, projpixel,
          l_gradx, l_grady, l_size, K, grad_threshold, inv_cov);
      //threshold small gradients
//      if (gradValid == false) {
//        row.result = -5;
//        continue;
//      }

      row.error = inv_cov * residual;

//      if (row.error  * row.error > residual_criteria){
////        std::cout<<row.error<<std::endl;
//        row.result = -4;
//        continue;
//      }

      row.result = 1;
    }
  }
  }

void Tracking::trackEachObject(ObjectPointer& objectptr,
                               const float4 k,
                               const Matrix4& T_w_c2,
                               const Matrix4& T_w_c1,
                               const float* f_l_I,
                               const float* f_l_D){

  bool tracked = false;
  //object pose in the last frame
  const Matrix4 T_wo1 = objectptr->volume_pose_;
  const float3 * w_V_m0 = objectptr->m_vertex;
  const float3 * w_N_m0 = objectptr->m_normal;

  Matrix4 T_c1_o1 = inverse(T_w_c1) * T_wo1;
  Matrix4 T_c2_o2 = inverse(T_w_c2) * T_wo1;


  if (!use_virtual_camera_){

    if (using_RGB_ && (!use_live_depth_only_)){
      for (unsigned int i = 0; i < GN_opt_iterations_.size(); ++i) {
        Matrix4 invK = getInverseCameraMatrix(k_ / float(1 << i));
        if (i == 0){
          vertex2depth(r_D_render_[0], w_V_m0, imgSize_, T_w_c1);
          depth2vertexKernel(r_Vertex_render_[0], r_D_render_[0],
                             localImgSize_[0], invK);
          //memcpy(r_Vertex_[0], model_vertex,
          //       sizeof(float3) * localImgSize_[0].x * localImgSize_[0].y);
        }
        else{
          //using the rendered depth
          halfSampleRobustImageKernel(r_D_render_[i], r_D_render_[i - 1],
                                      localImgSize_[i-1], e_delta * 3, 1);
          depth2vertexKernel(r_Vertex_render_[i], r_D_render_[i],
                             localImgSize_[i], invK);

        }
      }
    }

    Matrix4 pose_update;
    Matrix4 previous_pose = T_c2_o2;

    //coarse-to-fine iteration
    for (int level = GN_opt_iterations_.size() - 1; level >= 0; --level) {
      float previous_error = INFINITY;

      for (int i = 0; i < GN_opt_iterations_[level]; ++i) {
      if (using_ICP_){
        track_Obj_ICP_Kernel(trackingResult_[level], imgSize_,
                             l_vertex_[level], l_normal_[level],
                             localImgSize_[level],  w_V_m0, w_N_m0, imgSize_,
                             T_c2_o2, T_c1_o1, T_wo1, k,
                             dist_threshold,
                             normal_threshold, icp_cov_pyramid_[level],
                             outlier_mask_[level]);
      }


        if (using_RGB_) {
          track_obj_RGB_kernel(trackingResult_[level] + imgSize_.x * imgSize_.y,
                               imgSize_, r_Vertex_render_[level],
                               r_Vertex_live_[level], r_I_[level],
                               localImgSize_[level], l_I_[level], localImgSize_[level],
                               l_gradx_[level], l_grady_[level],
                               T_c2_o2, T_c1_o1,
                               getCameraMatrix(k_ / (1 << level)),
                               rgb_tracking_threshold_[level],
                               mini_gradient_magintude_[level],
                               sigma_bright_, prev_outlier_mask_[level]);
        }


        if (using_ICP_ && (!using_RGB_)){
          reduceKernel(reductionoutput_[level], trackingResult_[level],
                       imgSize_, localImgSize_[level]);
        }

        if ((!using_ICP_) && using_RGB_){
          reduceKernel(reductionoutput_[level],
                       trackingResult_[level]+ imgSize_.x * imgSize_.y,
                       imgSize_, localImgSize_[level]);
        }

        if (using_ICP_ && using_RGB_){
          reduceKernel(reductionoutput_[level], trackingResult_[level],
                       imgSize_, localImgSize_[level]);
        }


        const float current_error =
            reductionoutput_[level][0]/reductionoutput_[level][28];
//        std::cout<< "Level " << level << ", Iteration " << i << ", Error: " << current_error<<std::endl;

        if (reductionoutput_[level][28] == 0){
          break;
        }

        if (current_error > (previous_error )){
          T_c2_o2 = previous_pose;
//          std::cout<< "Level " << level << " Iteration " << i << " - Cost increase from " << previous_error << " to " <<
//                   current_error<<std::endl;
          break;
        }
        previous_error = current_error;

        if (solvePoseKernel(pose_update, reductionoutput_[level],
                            icp_threshold)) {
          break;
        }

        previous_pose = T_c2_o2;
        T_c2_o2 = pose_update * previous_pose;
      }
    }
    objectptr->volume_pose_ = T_w_c2* T_c2_o2;
    objectptr->virtual_camera_pose_ = T_wo1 * inverse(T_c2_o2);
  }
  else{

    Matrix4 T_wc1_v = T_w_c1;
    trackLiveFrame(T_wc1_v, T_w_c1, k, w_V_m0, w_N_m0);

    objectptr->volume_pose_ =  T_w_c1 *  inverse(T_wc1_v) * T_wo1;
    objectptr->virtual_camera_pose_ = T_wc1_v;

  }

}



bool Tracking::checkPose(Matrix4 & pose, const Matrix4& oldPose){
  bool checked=  checkPoseKernel(pose, oldPose, reductionoutput_[0], imgSize_,
                                 track_threshold);
  if (checked == false){
    std::cout<<"pose tracking is wrong, getting back to old pose"<<std::endl;
  }
  return checked;
//  return true;
}

TrackData* Tracking::getTrackingResult(){
  //if (!using_ICP_) return trackingResult_[0] + imgSize_.x * imgSize_.y;
  //else
    return  trackingResult_[0];
}

void Tracking::obtainErrorParameters(const Dataset& dataset) {

//read from yaml file
  //  try {
  //    cv::FileStorage fNode(filename, cv::FileStorage::READ);
  //  }
  //  catch (...) {
  //    assert(0 && "YAML file not parsed correctly.");
  //  }
  //
  //  cv::FileStorage fNode(filename, cv::FileStorage::READ);
  //
  //  if (!fNode.isOpened()) {
  //    assert(0 && "YAML file was not opened.");
  //  }
  //
  //  sigma_b = fNode["sigma_brightness"];
  //  sigma_disp = fNode["sigma_disparity"];
  //  sigma_xy = fNode["sigma_xy"];
  //  baseline = fNode["baseline"];
  //  focal = fNode["focal_length"];

//manual setting
  sigma_disparity_ = 1.0; //5.5
  sigma_xy_ = 1.0; //5.5

  if (dataset == Dataset::zr300){//zr300
    sigma_bright_ = 1.0f;
    baseline_ = 0.07f;
    focal_ = 617.164;
  }
  if (dataset == Dataset::asus){//asus
    baseline_ = 0.6;
    focal_ = 580.0;
  }
  if (dataset == Dataset::icl_nuim){//ICL-NUIM
    sigma_bright_ = sqrtf(100.0)/255.0;
    baseline_ = 0.075;
    focal_ = 481.2;
  }
  if (dataset == Dataset::tum_rgbd){//TUM datasets
    sigma_bright_ = sqrtf(100.0)/255.0;
    baseline_ = 0.075;
    focal_ = 525.0;
  }
}


void Tracking::buildPyramid(const float* l_I, const float* l_D){

  //half sample the coarse layers for input/reference rgb-d frames
  for (unsigned int i = 0; i < GN_opt_iterations_.size(); ++i) {
    if (i == 0){
      // memory copy the first layer
//      memcpy(l_I_[0], l_I, sizeof(float)*imgSize_.x*imgSize_.y);
//      memcpy(l_D_[0], l_D, sizeof(float)*imgSize_.x*imgSize_.y);
      // bilateral filtering the input depth
      bilateralFilterKernel(l_I_[0], l_I, imgSize_, gaussian, 0.1f, 2);
      bilateralFilterKernel(l_D_[0], l_D, imgSize_, gaussian, 0.1f, 2);
    }
    else{
      halfSampleRobustImageKernel(l_D_[i], l_D_[i - 1], localImgSize_[i-1],
                                  e_delta * 3, 1);
      halfSampleRobustImageKernel(l_I_[i], l_I_[i - 1],localImgSize_[i-1],
                                  e_delta * 3, 1);
      //using the rendered depth
      /*
      if (using_RGB_ && use_rendered_depth_){
        halfSampleRobustImageKernel(r_D_[i], r_D_[i - 1], localImgSize_[i-1],
                                    e_delta * 3, 1);
      }
       */
    }

    // prepare the 3D information from the input depth maps
    Matrix4 invK = getInverseCameraMatrix(k_ / float(1 << i));
    depth2vertexKernel(l_vertex_[i], l_D_[i], localImgSize_[i], invK);
    if(k_.y < 0)
      vertex2normalKernel<FieldType, true>(l_normal_[i], l_vertex_[i],
                                           localImgSize_[i]);
    else
      vertex2normalKernel<FieldType, false>(l_normal_[i], l_vertex_[i],
                                            localImgSize_[i]);
    if (using_RGB_){
      gradientsKernel(l_gradx_[i], l_grady_[i], l_I_[i], localImgSize_[i]);
      //depth2vertexKernel(r_Vertex_[i], r_D_[i], localImgSize_[i], invK);
      //comment out: rendered case: done in trackLive frame;
      //direct depth case, performed in last memcpy
    }
  }
}

/*
void Tracking::buildPreviousPyramid(const float* r_I, const float* r_D){

  //half sample the coarse layers for input/reference rgb-d frames
  for (unsigned int i = 0; i < GN_opt_iterations_.size(); ++i) {
    if (i == 0){
      // bilateral filtering the input depth
      bilateralFilterKernel(r_I_[0], r_I, imgSize_, gaussian, 0.1f, 2);
      bilateralFilterKernel(r_D_[0], r_D, imgSize_, gaussian, 0.1f, 2);
    }
    else{
      halfSampleRobustImageKernel(r_D_[i], r_D_[i - 1], localImgSize_[i-1],
                                  e_delta * 3, 1);
      halfSampleRobustImageKernel(r_I_[i], r_I_[i - 1],localImgSize_[i-1],
                                  e_delta * 3, 1);
    }

    // prepare the 3D information from the input depth maps
    Matrix4 invK = getInverseCameraMatrix(k_ / float(1 << i));
    depth2vertexKernel(r_Vertex_[i], r_D_[i], localImgSize_[i], invK);

  }
}
*/

void Tracking::calc_icp_cov(float3* conv_icp, const uint2 size,
                            const float* depth_image, const Matrix4& K,
                            const float sigma_xy, const float sigma_disp,
                            const float baseline)
{
  unsigned int pixely, pixelx;
  const float focal = K.data[0].x;
#pragma omp parallel for shared(conv_icp), private(pixelx, pixely)
  for (pixely = 0; pixely < size.y; pixely++) {
    for (pixelx = 0; pixelx < size.x; pixelx++) {
      float3 &cov = conv_icp[pixelx + pixely * size.x];

      //get depth
      float depth = depth_image[pixelx + pixely * size.x];

      //standard deviations
      cov.x = (depth / focal) * sigma_xy;
      cov.y = (depth / focal) * sigma_xy;
      cov.z = (depth * depth * sigma_disp) / (focal * baseline);

      //square to get the variances
      cov.x = cov.x * cov.x;
      cov.y = cov.y * cov.y;
      cov.z = cov.z * cov.z;
    }
  }
}



void Tracking::track_ICP_Kernel(TrackData *output,
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
                                const float3 *icp_cov_layer) {

  cv::Mat outlier_mask(cv::Size(inSize.x, inSize.y), CV_8UC1, cv::Scalar(0));
//  deprecated, using the same name function below to enable using mask
  track_ICP_Kernel(output, jacobian_size,
                   inVertex, inNormal, inSize,
                   refVertex, refNormal, refSize,
                   Ttrack, view,
                   dist_threshold, normal_threshold,
                   icp_cov_layer,
                   outlier_mask);
}


void Tracking::trackRGB(TrackData* output, const uint2 jacobian_size,
                        const float3 *r_vertices_render,
                        const float3 *r_vertices_live, const float* r_image,
                        uint2 r_size, const float* l_image, uint2 l_size,
                        const float * l_gradx, const float * l_grady,
                        const Matrix4& T_w_r, const Matrix4& T_w_l,
                        const Matrix4& K, const float residual_criteria,
                        const float grad_threshold, const float sigma_bright) {

  cv::Mat outlier_mask(cv::Size(r_size.x, r_size.y), CV_8UC1, cv::Scalar(0));

//  deprecated, using the same name function below to enable using mask
  trackRGB(output, jacobian_size,
           r_vertices_render, r_vertices_live, r_image, r_size,
           l_image, l_size,
           l_gradx, l_grady,
           T_w_r, T_w_l, K,
           residual_criteria, grad_threshold,
           sigma_bright, outlier_mask);
}


void Tracking::track_ICP_Kernel(TrackData *output,
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
                                const cv::Mat& outlier_mask) {
  TICK();
  uint2 pixel = make_uint2(0, 0);
  unsigned int pixely, pixelx;
#pragma omp parallel for \
      shared(output), private(pixel, pixelx, pixely)
  for (pixely = 0; pixely < inSize.y; pixely++) {
    for (pixelx = 0; pixelx < inSize.x; pixelx++) {
      pixel.x = pixelx;
      pixel.y = pixely;

      TrackData &row = output[pixel.x + pixel.y * jacobian_size.x];

      if (outlier_mask.at<uchar>(pixely, pixelx) != 0) {
        row.result = -6;
        continue;
      }

      if (inNormal[pixel.x + pixel.y * inSize.x].x == INVALID) {
        row.result = -1;
        continue;
      }

      const float3 projectedVertex = Ttrack
          * inVertex[pixel.x + pixel.y * inSize.x];
      const float3 projectedPos = view * projectedVertex;
//      const float2 projPixel = make_float2(
//          projectedPos.x / projectedPos.z + 0.5f,
//          projectedPos.y / projectedPos.z + 0.5f);
//      if (projPixel.x < 0 || projPixel.x > refSize.x - 1
//          || projPixel.y < 0 || projPixel.y > refSize.y - 1) {
//        row.result = -2;
//        continue;
//      }

      const float2 projPixel = make_float2(
          projectedPos.x / projectedPos.z,
          projectedPos.y / projectedPos.z);

      if (projPixel.x < 1 || projPixel.x > refSize.x - 2
          || projPixel.y < 1 || projPixel.y > refSize.y - 2 ||
          std::isnan(projPixel.x) || std::isnan(projPixel.y)) {
        row.result = -2;
        continue;
      }

      const uint2 refPixel = make_uint2(projPixel.x, projPixel.y);
      const float3 referenceNormal = refNormal[refPixel.x
          + refPixel.y * refSize.x];
//      const float3 referenceNormal = bilinear_interp(refNormal, refSize, projPixel);
      if (referenceNormal.x == INVALID) {
        row.result = -3;
        continue;
      }

      const float3 diff = refVertex[refPixel.x + refPixel.y * refSize.x]
          - projectedVertex;
//      const float3 diff = bilinear_interp(refVertex, refSize, projPixel) - projectedVertex;
      const float3 projectedNormal = rotate(Ttrack, inNormal[pixel.x + pixel.y * inSize.x]);

      if (length(diff) > dist_threshold) {
        row.result = -4;
        continue;
      }
      if (dot(projectedNormal, referenceNormal) < normal_threshold) {
        row.result = -5;
        continue;
      }
      row.result = 1;

      //calculate the inverse of covariance as weights
      const float3 P = icp_cov_layer[pixel.x + pixel.y * inSize.x];
      const float sigma_icp = referenceNormal.x * referenceNormal.x * P.x
          + referenceNormal.y * referenceNormal.y * P.y
          + referenceNormal.z * referenceNormal.z * P.z;
      const float inv_cov = sqrtf(1.0 / sigma_icp);

      row.error = inv_cov * dot(referenceNormal, diff);
      ((float3 *) row.J)[0] = inv_cov * referenceNormal;
      ((float3 *) row.J)[1] =
          inv_cov * cross(projectedVertex, referenceNormal);
    }
  }
  TOCK("trackKernel", inSize.x * inSize.y);
}


void Tracking::trackRGB(TrackData* output, const uint2 jacobian_size,
                        const float3 *r_vertices_render,
                        const float3 *r_vertices_live, const float* r_image,
                        uint2 r_size, const float* l_image, uint2 l_size,
                        const float * l_gradx, const float * l_grady,
                        const Matrix4& T_w_r, const Matrix4& T_w_l,
                        const Matrix4& K, const float residual_criteria,
                        const float grad_threshold, const float sigma_bright,
                        const cv::Mat& outlier_mask) {
  TICK();
  uint2 r_pixel = make_uint2(0, 0);
  unsigned int r_pixely, r_pixelx;
//  static const float sqrt_w = sqrtf(w);
#pragma omp parallel for shared(output),private(r_pixel,r_pixelx,r_pixely)
  for (r_pixely = 0; r_pixely < r_size.y; r_pixely++) {
    for (r_pixelx = 0; r_pixelx < r_size.x; r_pixelx++) {
      r_pixel.x = r_pixelx;
      r_pixel.y = r_pixely;

      TrackData & row = output[r_pixel.x + r_pixel.y * jacobian_size.x];

      if (outlier_mask.at<uchar>(r_pixely, r_pixelx) != 0) {
        row.result = -6;
        continue;
      }

      const int r_index = r_pixel.x + r_pixel.y * r_size.x;
      float3 r_vertex_render = r_vertices_render[r_index];
      const float3 r_vertex_live = r_vertices_live[r_index];

      //if rendered depth is not available
      if ((r_vertex_render.z <= 0.f) || (r_vertex_render.z == INVALID)) {
        //if live depth is not availvle too =>depth error
//        if (r_vertex_live.z <= 0.f ||r_vertex_live.z == INVALID) {
        row.result = -1;
        continue;
//        }

        /*else{
//          if live depth is availvle, use live depth instead
//          would introduce occlusion however
          r_vertex_render = r_vertex_live;
        }*/
      }

      //if the difference between rendered and live depth is too large =>occlude
      if (length(r_vertex_render - r_vertex_live) > occluded_depth_diff_){
        //not in the case that no live depth
        if (r_vertex_live.z > 0.f){
          row.result = -3;
//          std::cout<<r_vertex_render.z <<" "<<r_vertex_live.z<<std::endl;
          continue;
        }
      }


      float3 l_vertex, w_vertex;
      float2 proj_pixel = warp(l_vertex, w_vertex, T_w_r, T_w_l, K, r_vertex_render);
      if (proj_pixel.x < 1  || proj_pixel.x > l_size.x - 2
          || proj_pixel.y < 1 || proj_pixel.y > l_size.y - 2 ||
          std::isnan(proj_pixel.x) || std::isnan(proj_pixel.y)) {
        row.result = -2;
        continue;
      }


      const float residual = rgb_residual(r_image, r_pixel, r_size, l_image, proj_pixel, l_size);

      const float inv_cov = 1.0/sigma_bright;

      bool gradValid = rgb_jacobian(row.J, l_vertex, w_vertex, T_w_l, proj_pixel, l_gradx, l_grady, l_size, K,
                                    grad_threshold, inv_cov);
      //threshold small gradients
//      if (gradValid == false) {
//        row.result = -5;
//        continue;
//      }
      row.error = inv_cov * residual;

//      if (row.error  * row.error > residual_criteria){
////        std::cout<<row.error<<std::endl;
//        row.result = -4;
//        continue;
//      }

      row.result = 1;
    }
  }
}


float2 Tracking::warp(float3& l_vertex, float3& w_vertex, const Matrix4& T_w_r,
                      const Matrix4& T_w_l, const Matrix4& K, const float3& r_vertex) {
  w_vertex = T_w_r * r_vertex;
  l_vertex = inverse(T_w_l) * w_vertex;
  const float3 proj_vertex = rotate(K, l_vertex);
  return make_float2(proj_vertex.x/proj_vertex.z,
                     proj_vertex.y/proj_vertex.z);
}

bool Tracking::rgb_jacobian(float J[6], const float3& l_vertex,
                            const float3& w_vertex, const Matrix4& T_w_l,
                            const float2& proj_pixel,  const float* l_gradx,
                            const float* l_grady, const uint2& l_size,
                            const Matrix4& K, const float grad_threshold, const float weight) {

  const float gradx = bilinear_interp(l_gradx, l_size, proj_pixel);
  const float grady = bilinear_interp(l_grady, l_size, proj_pixel);
  const float grad_mag = length(make_float2(gradx, grady));

  if (grad_mag < grad_threshold) return false;

  const float fx = K.data[0].x;
  const float fy = K.data[1].y;

  float3 Jtrans = (1.f / l_vertex.z) * make_float3(gradx * fx, grady * fy,
                                                   -(gradx * l_vertex.x * fx + grady * l_vertex.y * fy) / l_vertex.z);

  Jtrans = -1.f * rotate(Jtrans, transpose(T_w_l));
  float3 Jrot = cross(w_vertex, Jtrans);

  /* Omitting the -1.f factor because in reduceKernel JTe is not
   * multiplied by -1.f. */
  ((float3 *) J)[0] = /* -1.f */ weight * Jtrans;
  ((float3 *) J)[1] = /* -1.f */ weight * Jrot;

  return true;
}

bool Tracking::obj_rgb_jacobian(float J[6], const float3& l_vertex,
                                const float2& proj_pixel, const float* l_gradx,
                                const float* l_grady, const uint2& l_size,
                                const Matrix4& K, const float grad_threshold,
                                const float weight) {

  const float gradx = bilinear_interp(l_gradx, l_size, proj_pixel);
  const float grady = bilinear_interp(l_grady, l_size, proj_pixel);
  const float grad_mag = length(make_float2(gradx, grady));

  if (grad_mag < grad_threshold) return false;

  const float fx = K.data[0].x;
  const float fy = K.data[1].y;

  float3 Jtrans = (1.f / l_vertex.z) * make_float3(gradx * fx, grady * fy,
                                                   -(gradx * l_vertex.x * fx + grady * l_vertex.y * fy) / l_vertex.z);

  float3 Jrot = cross(l_vertex, Jtrans);

  /* Omitting the -1.f factor because in reduceKernel JTe is not
   * multiplied by -1.f. */
  ((float3 *) J)[0] = /* -1.f */ weight * Jtrans;
  ((float3 *) J)[1] = /* -1.f */ weight * Jrot;

  return true;
}

float Tracking::rgb_residual(const float* r_image, const uint2& r_pixel,
                             const uint2& r_size, const float* l_image,
                             const float2& proj_pixel, const uint2& l_size)
{
  float l_interpolated = bilinear_interp(l_image, l_size, proj_pixel);
  return (r_image[r_pixel.x + r_pixel.y * r_size.x] - l_interpolated);
}




void Tracking::reduceKernel(float *out, TrackData *J, const uint2 Jsize,
                            const uint2 size) {
  TICK();
  int blockIndex;
  for (blockIndex = 0; blockIndex < 8; ++blockIndex)
    reduce(blockIndex, out, J, Jsize, size, this->stack_);

  TooN::Matrix<8, 32, float, TooN::Reference::RowMajor> values(out);
  for (int j = 1; j < 8; ++j) {
    values[0] += values[j];
    //std::cerr << "REDUCE ";for(int ii = 0; ii < 32;ii++)
    //std::cerr << values[0][ii] << " ";
    //std::cerr << "\n";
  }
  TOCK("reduceKernel", 512);
}

void Tracking::reduce(int blockIndex, float *out, TrackData *J, const uint2
Jsize, const uint2 size, const int stack) {
  float *sums = out + blockIndex * 32;

  for (uint i = 0; i < 32; ++i)
    sums[i] = 0;
  float sums0, sums1, sums2, sums3, sums4, sums5, sums6, sums7, sums8, sums9,
      sums10, sums11, sums12, sums13, sums14, sums15, sums16, sums17,
      sums18, sums19, sums20, sums21, sums22, sums23, sums24, sums25,
      sums26, sums27, sums28, sums29, sums30, sums31;
  sums0 = 0.0f;
  sums1 = 0.0f;
  sums2 = 0.0f;
  sums3 = 0.0f;
  sums4 = 0.0f;
  sums5 = 0.0f;
  sums6 = 0.0f;
  sums7 = 0.0f;
  sums8 = 0.0f;
  sums9 = 0.0f;
  sums10 = 0.0f;
  sums11 = 0.0f;
  sums12 = 0.0f;
  sums13 = 0.0f;
  sums14 = 0.0f;
  sums15 = 0.0f;
  sums16 = 0.0f;
  sums17 = 0.0f;
  sums18 = 0.0f;
  sums19 = 0.0f;
  sums20 = 0.0f;
  sums21 = 0.0f;
  sums22 = 0.0f;
  sums23 = 0.0f;
  sums24 = 0.0f;
  sums25 = 0.0f;
  sums26 = 0.0f;
  sums27 = 0.0f;
  sums28 = 0.0f;
  sums29 = 0.0f;
  sums30 = 0.0f;
  sums31 = 0.0f;

// comment me out to try coarse grain parallelism
#pragma omp parallel for reduction(+:sums0, sums1, sums2, sums3, sums4, sums5, sums6, sums7, sums8, sums9, sums10, sums11, sums12, sums13, sums14, sums15, sums16, sums17, sums18, sums19, sums20, sums21, sums22, sums23, sums24, sums25, sums26, sums27, sums28, sums29, sums30, sums31)
  for (uint y = blockIndex; y < size.y; y += 8) {
    for (uint x = 0; x < size.x; x++) {
      for (unsigned int k = 0; k < stack; ++k) {

        const unsigned stacked_offset = k * (Jsize.x * Jsize.y);
        const TrackData &row = J[stacked_offset + (x + y * Jsize.x)]; // ...
        if (row.result < 1) {
          // accesses sums[28..31]
          /*(sums+28)[1]*/sums29 += row.result == -4 ? 1 : 0;
          /*(sums+28)[2]*/sums30 += row.result == -5 ? 1 : 0;
          /*(sums+28)[3]*/sums31 += row.result > -4 ? 1 : 0;

          continue;
        }

        float irls_weight = calc_robust_weight(row.error, robustWeight_);

        // Error part
        /*sums[0]*/sums0 += calc_robust_residual(row.error, robustWeight_);

        // JTe part
        /*for(int i = 0; i < 6; ++i)
          sums[i+1] += row.error * row.J[i];*/
        sums1 += irls_weight * row.error * row.J[0];
        sums2 += irls_weight * row.error * row.J[1];
        sums3 += irls_weight * row.error * row.J[2];
        sums4 += irls_weight * row.error * row.J[3];
        sums5 += irls_weight * row.error * row.J[4];
        sums6 += irls_weight * row.error * row.J[5];

        // JTJ part, unfortunatly the double loop is not unrolled well...
        /*(sums+7)[0]*/sums7 += irls_weight * row.J[0] * row.J[0];
        /*(sums+7)[1]*/sums8 += irls_weight * row.J[0] * row.J[1];
        /*(sums+7)[2]*/sums9 += irls_weight * row.J[0] * row.J[2];
        /*(sums+7)[3]*/sums10 += irls_weight * row.J[0] * row.J[3];

        /*(sums+7)[4]*/sums11 += irls_weight * row.J[0] * row.J[4];
        /*(sums+7)[5]*/sums12 += irls_weight * row.J[0] * row.J[5];

        /*(sums+7)[6]*/sums13 += irls_weight * row.J[1] * row.J[1];
        /*(sums+7)[7]*/sums14 += irls_weight * row.J[1] * row.J[2];
        /*(sums+7)[8]*/sums15 += irls_weight * row.J[1] * row.J[3];
        /*(sums+7)[9]*/sums16 += irls_weight * row.J[1] * row.J[4];

        /*(sums+7)[10]*/sums17 += irls_weight * row.J[1] * row.J[5];

        /*(sums+7)[11]*/sums18 += irls_weight * row.J[2] * row.J[2];
        /*(sums+7)[12]*/sums19 += irls_weight * row.J[2] * row.J[3];
        /*(sums+7)[13]*/sums20 += irls_weight * row.J[2] * row.J[4];
        /*(sums+7)[14]*/sums21 += irls_weight * row.J[2] * row.J[5];

        /*(sums+7)[15]*/sums22 += irls_weight * row.J[3] * row.J[3];
        /*(sums+7)[16]*/sums23 += irls_weight * row.J[3] * row.J[4];
        /*(sums+7)[17]*/sums24 += irls_weight * row.J[3] * row.J[5];

        /*(sums+7)[18]*/sums25 += irls_weight * row.J[4] * row.J[4];
        /*(sums+7)[19]*/sums26 += irls_weight * row.J[4] * row.J[5];

        /*(sums+7)[20]*/sums27 += irls_weight * row.J[5] * row.J[5];

        // extra info here
        /*(sums+28)[0]*/sums28 += 1;

      }
    }
  }
  sums[0] = sums0;
  sums[1] = sums1;
  sums[2] = sums2;
  sums[3] = sums3;
  sums[4] = sums4;
  sums[5] = sums5;
  sums[6] = sums6;
  sums[7] = sums7;
  sums[8] = sums8;
  sums[9] = sums9;
  sums[10] = sums10;
  sums[11] = sums11;
  sums[12] = sums12;
  sums[13] = sums13;
  sums[14] = sums14;
  sums[15] = sums15;
  sums[16] = sums16;
  sums[17] = sums17;
  sums[18] = sums18;
  sums[19] = sums19;
  sums[20] = sums20;
  sums[21] = sums21;
  sums[22] = sums22;
  sums[23] = sums23;
  sums[24] = sums24;
  sums[25] = sums25;
  sums[26] = sums26;
  sums[27] = sums27;
  sums[28] = sums28;
  sums[29] = sums29;
  sums[30] = sums30;
  sums[31] = sums31;

}

bool Tracking::solvePoseKernel(Matrix4 & pose_update, const float * output,
                               float icp_threshold) {
  bool res = false;
  TICK();
  // Update the pose regarding the tracking result
  TooN::Matrix<8, 32, const float, TooN::Reference::RowMajor> values(output);
  TooN::Vector<6> x = solve(values[0].slice<1, 27>());
  TooN::SE3<> delta(x);
  pose_update = toMatrix4(delta);

  // Return validity test result of the tracking
  if ( (norm(x) < icp_threshold) && (std::sqrt(output[0]/output[28]) < 2e-3) ){
//    std::cout<<"updating done, jump of normal equation solving iteration"<<std::endl;
    res = true;}

  //    std::cout<<"updating pose, with pertubation: "<<norm(x)<<" residual: "<<(std::sqrt(output[0]/output[28]))
//             <<std::endl;

  TOCK("updatePoseKernel", 1);
  return res;
}

bool Tracking::checkPoseKernel(Matrix4 & pose, const Matrix4& oldPose,
                               const float* output, const uint2 imageSize,
                               const float track_threshold) {

  // Check the tracking result, and go back to the previous camera position if necessary

  TooN::Matrix<8, 32, const float, TooN::Reference::RowMajor> values(output);

  bool is_residual_high =(std::sqrt(values(0, 0) / values(0, 28)) > 2e-1);
  bool is_trackPoints_few = (values(0, 28) / (imageSize.x * imageSize.y) < track_threshold);

  if ( is_residual_high|| is_trackPoints_few) {
    pose = oldPose;

    //      std::cerr<<"tracking fails. residual: "<<(std::sqrt(values(0, 0) / values(0, 28) ))<<"pass? "<<is_residual_high<<" tracking "
//          "points: "<<(values(0, 28) / (imageSize.x * imageSize.y))<<"pass? "<<is_trackPoints_few<<std::endl;
    return false;
  } else {
    //      std::cout<<"pose updating checked"<<std::endl;
    return true;
  }

}


void Tracking::setRefImgFromCurr(){
  if (using_RGB_){
    for (int level = GN_opt_iterations_.size() - 1; level >= 0; --level) {
      memcpy(r_I_[level], l_I_[level],
             sizeof(float)*localImgSize_[level].x*localImgSize_[level].y);
        memcpy(r_D_live_[level], l_D_[level],
               sizeof(float)*localImgSize_[level].x*localImgSize_[level].y);
        memcpy(r_Vertex_live_[level], l_vertex_[level],
               sizeof(float3) * localImgSize_[level].x * localImgSize_[level].y);
    }
    prev_outlier_mask_ = outlier_mask_;
  }
  outlier_mask_.clear();
}


/*
 * backward rendering
 * warp the live image to the ref image position based on the T_w_l and T_w_r
 * and then calculate the residual image between warped image and ref image
 */


void Tracking::warp_to_residual(cv::Mat& warped_image, cv::Mat& residual,
                                const float* l_image, const float* ref_image,
                                const float3* r_vertex, const Matrix4& T_w_l,
                                const Matrix4& T_w_r, const Matrix4& K, const uint2 outSize){

  residual = cv::Mat::zeros(cv::Size(outSize.x, outSize.y), CV_8UC1);
  warped_image = cv::Mat::zeros(cv::Size(outSize.x, outSize.y), CV_8UC1);

  unsigned int y;
#pragma omp parallel for \
        shared(warped_image, residual), private(y)
  for (y = 0; y < outSize.y; y++)
    for (unsigned int x = 0; x < outSize.x; x++) {
      const float3 vertex = r_vertex[x + y*outSize.x];
      if (vertex.z == 0.f ||vertex.z == INVALID) {
        continue;
      }

      float3 l_vertex, w_vertex;
      float2 projpixel = warp(l_vertex, w_vertex, T_w_r, T_w_l, K,
                              vertex);
      if (projpixel.x < 1  || projpixel.x > outSize.x - 2
          || projpixel.y < 1 || projpixel.y > outSize.y - 2) {
        continue;
      }

      //the corresponding pixel of (x, y) on the warped_image is the
      // projpixel on the input_image
      float interpolated = bilinear_interp(l_image, outSize, projpixel);
      warped_image.at<uchar>(y, x) = 255.0 * interpolated;

      float ref_intensity = ref_image[x + y*outSize.x];
      residual.at<uchar>(y, x) = 255.0 * fabs(ref_intensity - interpolated);
    }
}

void Tracking::dump_residual(TrackData* res_data, const uint2 inSize){
  std::ofstream residual_file;
  residual_file.open("residual.csv");
  unsigned int pixely, pixelx;
  for (pixely = 0; pixely < inSize.y; pixely++) {
    for (pixelx = 0; pixelx < inSize.x; pixelx++) {
      int id = pixelx + pixely * inSize.x;
      if (res_data[id].result < 1) continue;
      residual_file<<sq(res_data[id].error)<<",\n";
    }
  }
  residual_file.close();
}

float Tracking::calc_robust_residual(const float residual,
                                     const RobustW& robustfunction){
  double loss = 0.f;
  switch (robustfunction){
    case RobustW::noweight:
      loss = 0.5 * residual * residual;
      break;
    case RobustW::Huber:
      if (fabsf(residual) <= huber_k){
        loss = 0.5 * residual * residual;
        break;
      } else{
        loss = huber_k * (fabsf(residual) - 0.5 * huber_k);
        break;
      }
    case RobustW::Tukey:
      if (fabsf(residual) <= tukey_b){
        loss = tukey_b * tukey_b / 6.0 * (1.0 - pow(1.0 - sq(residual/tukey_b), 3));
        break;
      } else{
        loss = tukey_b * tukey_b / 6.0;
        break;
      }
    case RobustW::Cauchy:
      loss = 0.5 * cauchy_k * cauchy_k * log(1.0 + sq(residual/cauchy_k));
      break;
  }
  return static_cast<float>(loss);
}

float Tracking::calc_robust_weight(const float residual,
                                   const RobustW& robustfunction){
  double irls_weight = 1.0f;
  switch (robustfunction){
    case RobustW::noweight:
      irls_weight = 1.0f;
      break;
    case RobustW::Huber:
      if (fabsf(residual) <= huber_k){
        irls_weight = 1.0f;
        break;
      } else{
        irls_weight = huber_k/fabsf(residual);
        break;
      }
    case RobustW::Tukey:
      if (fabsf(residual) <= tukey_b){
        irls_weight = sq(1.0 - sq(residual)/sq(tukey_b));
        break;
      } else{
        irls_weight = 0;
        break;
      }
    case RobustW::Cauchy:
      irls_weight = 1/(1 + sq(residual/cauchy_k));
      break;
  }
  return static_cast<float>(irls_weight);
}


void Tracking::compute_residuals(const Matrix4& T_w_l,
                                 const Matrix4& T_w_r,
                                 const float3* w_refVertices_r,
                                 const float3* w_refNormals_r,
                                 const float3* w_refVertices_l,
                                 const bool computeICP,
                                 const bool computeRGB,
                                 const float threshold){

  //overuse the r_D_ and r_Vertex_ for rendedered live depth and vertices
  Matrix4 invK = getInverseCameraMatrix(k_);
  vertex2depth(l_D_ref_, w_refVertices_l, imgSize_, T_w_r);
  depth2vertexKernel(l_vertex_ref_, l_D_ref_, localImgSize_[0], invK);

  compute_residual_kernel(trackingResult_[0], imgSize_, T_w_r, T_w_l, k_,
                          r_I_[0], l_I_[0], l_vertex_[0], l_normal_[0],
                          w_refVertices_r, w_refNormals_r,  l_vertex_ref_,
                          computeICP, computeRGB, icp_cov_pyramid_[0],
                          sigma_bright_, threshold);
}

void Tracking::compute_residual_kernel(TrackData* res_data,
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
                                       const float threshold){
  TICK();
  uint2 pixel = make_uint2(0, 0);
  const Matrix4 K = getCameraMatrix(k);
  unsigned int pixely, pixelx;
#pragma omp parallel for \
      shared(res_data), private(pixel, pixelx, pixely)
  for (pixely = 0; pixely < img_size.y; pixely++) {
    for (pixelx = 0; pixelx < img_size.x; pixelx++) {
      pixel.x = pixelx;
      pixel.y = pixely;

      const int in_index = pixel.x + pixel.y * img_size.x;

      if (outlier_mask_[0].at<uchar>(pixel.y, pixel.x) != 0) {
        res_data[in_index].result = -6;
        res_data[in_index + img_size.x * img_size.y].result = -6;
        continue;
      }


      //ICP
      if(computeICP){
        TrackData& icp_row = res_data[in_index];
        compute_icp_residual_kernel(icp_row, in_index, img_size, T_w_l,
                                    T_w_r, K, l_vertices, l_normals,
                                    w_refVertices_r, w_refNormals_r,
                                    icp_cov_layer, threshold);
      }


      //RGB residual
      if (computeRGB){
        TrackData& rgb_row = res_data[in_index + img_size.x * img_size.y];
        compute_rgb_residual_kernel(rgb_row, pixel, img_size, T_w_l, T_w_r, K,
                                    r_image, l_image, l_refVertices_l,
                                    l_vertices, sigma_bright, threshold);
      }
    }
  }
}


void Tracking::compute_icp_residual_kernel(TrackData& icp_row,
                                           const int in_index,
                                           const uint2 img_size,
                                           const Matrix4& T_w_l,
                                           const Matrix4& T_w_r,
                                           const Matrix4& K,
                                           const float3* l_vertices,
                                           const float3* l_normals,
                                           const float3* w_refVertices_r,
                                           const float3* w_refNormals_r,
                                           const float3* icp_cov_layer,
                                           const float threshold){
  if (l_normals[in_index].x == INVALID) {
    icp_row.result = -1;
    return;
  }

  float3 w_vertex, r_vertex;
  const float3 l_vertex = l_vertices[in_index];
  float2 proj_pixel = warp(r_vertex, w_vertex, T_w_l, T_w_r, K, l_vertex);
  if (proj_pixel.x < 1 || proj_pixel.x > img_size.x - 2
      || proj_pixel.y < 1 || proj_pixel.y > img_size.y - 2) {
    icp_row.result = -2;
    return;
  }

  const uint2 ref_pixel = make_uint2(proj_pixel.x, proj_pixel.y);
  const int ref_index = ref_pixel.x + ref_pixel.y * img_size.x;
  const float3 w_refnormal_r = w_refNormals_r[ref_index];
  const float3 w_refvertex_r = w_refVertices_r[ref_index];

  if (w_refnormal_r.x == INVALID){
    icp_row.result = -3;
    return;
  }

  const float3 diff = w_refvertex_r - w_vertex;
  const float3 w_normal = rotate(T_w_l, l_normals[in_index]);

  if (length(diff) > dist_threshold) {
    icp_row.result = -4;
    return;
  }
  if (dot(w_normal, w_refnormal_r) < normal_threshold) {
    icp_row.result = -5;
    return;
  }


//calculate the inverse of covariance as weights
  const float3 P = icp_cov_layer[in_index];
  const double sigma_icp = w_refnormal_r.x * w_refnormal_r.x * P.x
      + w_refnormal_r.y * w_refnormal_r.y * P.y
      + w_refnormal_r.z * w_refnormal_r.z * P.z;
  const double inv_cov = sqrt(1.0 / sigma_icp);

  const double icp_error = inv_cov * dot(w_refnormal_r, diff);

  if (fabs(icp_error) > threshold){
    icp_row.result = -4;
    return;
  }

  icp_row.error = static_cast<float>(icp_error);
  icp_row.result = 1;

}


void Tracking::compute_rgb_residual_kernel(TrackData& rgb_row,
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
                                           const float threshold){

  const int in_index = in_pixel.x + in_pixel.y * img_size.x;
  float3 l_vertex_ref = l_refVertices_l[in_index];
  const float3 l_vertex_live = l_Vertices_l[in_index];

  if (l_vertex_ref.z <= 0.f ||l_vertex_ref.z == INVALID) {
    if (fill_in_missing_depth_rgb_residual_checking_){
      if (l_vertex_live.z <= 0.f ||l_vertex_live.z == INVALID) {
        rgb_row.result = -1;
        return;
      }
      else{
        l_vertex_ref = l_vertex_live;
      }
    }
    else{
      rgb_row.result = -1;
      return;
    }
  }

  if (length(l_vertex_ref - l_vertex_live) > occluded_depth_diff_){
    //not in the case that no live depth
    if (l_vertex_live.z > 0.f){
      rgb_row.result = -5;
      return;
    }
  }

  float3 r_vertex, w_vertex;
  float2 proj_pixel = warp(r_vertex, w_vertex, T_w_l, T_w_r, K, l_vertex_ref);
  if (proj_pixel.x < 1  || proj_pixel.x > img_size.x - 2
      || proj_pixel.y < 1 || proj_pixel.y > img_size.y - 2) {
    rgb_row.result = -2;
    return;
  }

  rgb_row.result = 1;
  const float residual = rgb_residual(l_image, in_pixel, img_size,
                                      r_image, proj_pixel, img_size);

  const double inv_cov = 1.0/sigma_bright;
  const double rgb_error = inv_cov * residual;

  if (fabs(rgb_error) > threshold){
    rgb_row.result = -4;
    return;
  }

  rgb_row.error = static_cast<float>(rgb_error);
  rgb_row.result = 1;
}


bool Tracking::trackLiveFrame(Matrix4& T_w_l,
                              const Matrix4& T_w_r,
                              const float4 k,
                              const float3* model_vertex,
                              const float3* model_normal,
                              const cv::Mat& mask){

//  T_w_l = T_w_r;  //initilize the new live pose

//  //rendering the reference depth information from model
  if (using_RGB_ && (!use_live_depth_only_)){
    for (unsigned int i = 0; i < GN_opt_iterations_.size(); ++i) {
      Matrix4 invK = getInverseCameraMatrix(k_ / float(1 << i));
      if (i == 0){
        vertex2depth(r_D_render_[0], model_vertex, imgSize_, T_w_r);
        depth2vertexKernel(r_Vertex_render_[0], r_D_render_[0],
                           localImgSize_[0], invK);

        //memcpy(r_Vertex_[0], model_vertex,
        //       sizeof(float3) * localImgSize_[0].x * localImgSize_[0].y);
      }
      else{
        //using the rendered depth
        halfSampleRobustImageKernel(r_D_render_[i], r_D_render_[i - 1],
                                    localImgSize_[i-1], e_delta * 3, 1);
        depth2vertexKernel(r_Vertex_render_[i], r_D_render_[i],
                           localImgSize_[i], invK);

      }
    }
  }

  std::vector<cv::Mat> outlier_masks;
  for (unsigned int i = 0; i < GN_opt_iterations_.size(); ++i) {
    cv::Mat outlier_mask;
    cv::Size mask_size = mask.size() / ((1 << i));
    cv::resize(mask, outlier_mask, mask_size);
    outlier_masks.push_back(outlier_mask);
  }

  Matrix4 pose_update;

  Matrix4 previous_pose = T_w_l;

  //coarse-to-fine iteration
  for (int level = GN_opt_iterations_.size() - 1; level >= 0; --level) {
    float previous_error = INFINITY;
    for (int i = 0; i < GN_opt_iterations_[level]; ++i) {
      if (using_ICP_){
        const Matrix4 projectReference = getCameraMatrix(k_) * inverse(T_w_r);
        track_ICP_Kernel(trackingResult_[level], imgSize_, l_vertex_[level],
                         l_normal_[level], localImgSize_[level], model_vertex,
                         model_normal, imgSize_, T_w_l, projectReference,
                         dist_threshold, normal_threshold,
                         icp_cov_pyramid_[level], outlier_masks[level]);
      }

      if (using_RGB_){
        //render reference image to live image -- opposite to the original function call
        if (use_live_depth_only_){
          trackRGB(trackingResult_[level] + imgSize_.x * imgSize_.y,
                   imgSize_, r_Vertex_live_[level],r_Vertex_live_[level],
                   r_I_[level], localImgSize_[level],
                   l_I_[level], localImgSize_[level], l_gradx_[level],
                   l_grady_[level], T_w_r, T_w_l, getCameraMatrix(k_ / (1 << level)),
                   rgb_tracking_threshold_[level],
                   mini_gradient_magintude_[level], sigma_bright_,
                   prev_outlier_mask_[level]);
        }
        else{
          trackRGB(trackingResult_[level] + imgSize_.x * imgSize_.y,
                   imgSize_, r_Vertex_render_[level],r_Vertex_live_[level],
                   r_I_[level], localImgSize_[level],
                   l_I_[level], localImgSize_[level], l_gradx_[level],
                   l_grady_[level], T_w_r, T_w_l, getCameraMatrix(k_ / (1 << level)),
                   rgb_tracking_threshold_[level],
                   mini_gradient_magintude_[level], sigma_bright_,
                   prev_outlier_mask_[level]);
        }
      }

      if (using_ICP_ && (!using_RGB_)){
        reduceKernel(reductionoutput_[level], trackingResult_[level],
                     imgSize_, localImgSize_[level]);
      }

      if ((!using_ICP_) && using_RGB_){
        reduceKernel(reductionoutput_[level],
                     trackingResult_[level]+ imgSize_.x * imgSize_.y,
                     imgSize_, localImgSize_[level]);
      }

      if (using_ICP_ && using_RGB_){
        reduceKernel(reductionoutput_[level], trackingResult_[level],
                     imgSize_, localImgSize_[level]);
      }


      const float current_error =
          reductionoutput_[level][0]/reductionoutput_[level][28];
//      std::cout<< "Level " << level << ", Iteration " << i << ", Error: " << current_error<<std::endl;

      if (current_error > (previous_error /*+ 1e-5f*/)){
//        std::cout<< "Level " << level << " Iteration " << i << " - Cost increase from " << previous_error << " to " <<
//                 current_error<<std::endl;
        if (step_back_in_GN_) {T_w_l = previous_pose;}
        break;
      }
      previous_error = current_error;

      if (solvePoseKernel(pose_update, reductionoutput_[level],
                          icp_threshold)) {
//        previous_pose = T_w_l;
//        T_w_l = pose_update * previous_pose;
        break;
      }

      previous_pose = T_w_l;
      T_w_l = pose_update * previous_pose;
//      printMatrix4("updated live pose", T_w_l);
    }
  }

//  check the pose issue
//  bool tracked = checkPoseKernel(T_w_l, T_w_r, reductionoutput_[0], imgSize_,
//                                 track_threshold);
  bool tracked = true;

  return tracked;
}