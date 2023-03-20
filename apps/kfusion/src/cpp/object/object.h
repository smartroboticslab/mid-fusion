/*
 * SPDX-FileCopyrightText: 2017-2019 Smart Robotics Lab, Imperial College London
 * SPDX-FileCopyrightText: 2017-2019 Binbin Xu
 * SPDX-License-Identifier: BSD-3-Clause
 */
 

#ifndef OFUSION_OBJECT_H
#define OFUSION_OBJECT_H

#include "vector_types.h"
#include <commons.h>
#include "math_utils.h"
#include <map>
#include <vector>
#include <memory>
#include <segmentation.h>
#include "../continuous/volume_instance.hpp"

#include "../bfusion/mapping_impl.hpp"
#include "../kfusion/mapping_impl.hpp"
#include "../bfusion/alloc_impl.hpp"
#include "../kfusion/alloc_impl.hpp"

#include "opencv2/opencv.hpp"

class Object;
typedef std::shared_ptr<Object> ObjectPointer;
typedef std::vector<ObjectPointer> ObjectList;
typedef ObjectList::iterator ObjectListIterator;


class Object{
 private:
//  volume
  int voxel_block_size_ = 8;
  float3 volume_size_ = make_float3(10);//dimensions=size
  uint3 volume_resol_ = make_uint3(1024);//resolutions
  float volume_step = min(volume_size_) / max(volume_resol_);
  bool isStatic_;

//  SegmentationResult;
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
  //initilisation
//  Object(const Matrix4& pose, const int id, const uint2 imgSize);

  Object(const int voxel_block_size, const float3& volume_size, const uint3&
  volume_resol, const Matrix4& pose, const Matrix4& virtual_T_w_c,
         const int class_id, const All_Prob_Vect& all_prob,
         const uint2 imgSize);

  void set_volume_size(const float3& volume_size){
    this->volume_size_ = volume_size;
  };

  float3 get_volume_size(){
    return this->volume_size_;
  };

  uint3 get_volume_resol(){
    return this->volume_resol_;
  };

  float get_voxel_size(){
    return volume_size_.x/volume_resol_.x;
  }

  float get_volume_step(){
    return volume_step;
  }

  /**
 * @brief integrate a depth image into volume, based on camera pose T_w_c
 * @param depthImage [in] input depth image
 * @param rgbImage [in] input rgb image
 * @param imgSize [in] image size
 * @param T_w_c [in] camera pose
 * @param mu [in] TSDF mu
 * @param k [in] intrinsic matrix
 * @param frame [in] frame number for bfusion integration
 */
  void integration_static_kernel(const float * depthImage, const float3*rgbImage,
                                 const uint2& imgSize, const Matrix4& T_w_c,
                                 const float mu, const float4& k, const uint frame);

  /**
* @brief integrate a depth image into volume, based on camera pose T_w_c, and
   * the estimated volume pose T_w_o
* @param depthImage [in] input depth image
* @param rgbImage [in] input rgb image
* @param mask [in] segmentation mask corresponding this volume instance
* @param imgSize [in] image size
* @param T_w_c [in] camera pose
* @param mu [in] TSDF mu
* @param k [in] intrinsic matrix
* @param frame [in] frame number for bfusion integration
*/
  void integrate_volume_kernel(const float * depthImage, const float3*rgbImage,
                               const cv::Mat& mask, const uint2& imgSize,
                               const Matrix4& T_w_c, const float mu,
                               const float4& k, const uint frame);

  void fuse_semantic_kernel(const SegmentationResult& segmentationResult,
                            const int instance_id);

  void fuseSemanticLabel(const instance_seg& input_seg);

  void refine_mask_use_motion(cv::Mat& mask, const bool use_icp,
                              const bool use_rgb);

  void set_static_state(const bool state);

  bool is_static() const;

  virtual ~Object();

 public:
  //octree-based volume representation
  Volume<FieldType> volume_;
  octlib::key_t* allocationList_ = nullptr;
  size_t reserved_ = 0;

  //vertex and normal belonging to this volumne
  float3 * m_vertex;
  float3 * m_normal;

  //vertex and normal before intergration, belonging to this volumne
  float3 * m_vertex_bef_integ;
  float3 * m_normal_bef_integ;

  //pose
  Matrix4 volume_pose_;

  //virtual camera pose
  Matrix4 virtual_camera_pose_;

  //labels
  int class_id_;
  int instance_label_;

  //semanticfusion
  All_Prob_Vect semanticfusion_ = All_Prob_Vect::Zero();
  int fused_time_;
  static const uint label_size_ = 80;

  //Tracking result for this frame
  TrackData* trackresult_;

  static const std::set<int> static_object;
  bool pose_saved_;

  static bool absorb_outlier_bg;
};




#endif //OFUSION_OBJECT_H
