/*
 * SPDX-FileCopyrightText: 2017-2019 Smart Robotics Lab, Imperial College London
 * SPDX-FileCopyrightText: 2017-2019 Binbin Xu
 * SPDX-License-Identifier: BSD-3-Clause
 */
 

#include "object.h"

//Object::Object(const Matrix4& pose, const int id, const uint2 imgSize)
//    : volume_pose_(pose), class_id_(id){
//  volume_.init(volume_resol_.x, volume_size_.x);
//  m_vertex = (float3*) calloc(
//      sizeof(float3) * imgSize.x * imgSize.y, 1);
//  m_normalmal = (float3*) calloc(
//      sizeof(float3) * imgSize.x * imgSize.y, 1);
//};

Object::Object(const int voxel_block_size, const float3& volume_size, const uint3&
volume_resol, const Matrix4& pose, const Matrix4& virtual_T_w_c, const int
class_id, const All_Prob_Vect&
all_prob, const uint2 imgSize)
    : voxel_block_size_(voxel_block_size), volume_size_(volume_size),
      volume_resol_(volume_resol), isStatic_(true), volume_pose_(pose),
      virtual_camera_pose_(virtual_T_w_c), class_id_(class_id),
      semanticfusion_(all_prob)
{
  volume_.init(volume_resol.x, volume_size.x);
  volume_step = min(volume_size_) / max(volume_resol_);
  m_vertex = (float3*) calloc(
      sizeof(float3) * imgSize.x * imgSize.y, 1);
  m_normal = (float3*) calloc(
      sizeof(float3) * imgSize.x * imgSize.y, 1);

  m_vertex_bef_integ = (float3*) calloc(
      sizeof(float3) * imgSize.x * imgSize.y, 1);
  m_normal_bef_integ = (float3*) calloc(
      sizeof(float3) * imgSize.x * imgSize.y, 1);

  //semanticfusion
  fused_time_ = 1;

  //tracking result
  trackresult_ = (TrackData*) calloc(sizeof(TrackData) * imgSize.x * imgSize.y * 2, 1);
  pose_saved_ = false;
}

bool Object::absorb_outlier_bg = false;

Object::~Object(){
  this->volume_.release();
  if(allocationList_) delete(allocationList_);
  free(m_vertex);
  free(m_normal);
  free(m_vertex_bef_integ);
  free(m_normal_bef_integ);
  free(trackresult_);
};

void Object::integration_static_kernel(const float * depthImage,
                                       const float3*rgbImage,
                                       const uint2& imgSize, const Matrix4& T_w_c,
                                       const float mu, const float4& k,
                                       const uint frame){
  const float &voxelsize =  this->get_voxel_size();
  int num_vox_per_pix = volume_._size/((VoxelBlock<FieldType>::side) *voxelsize);
  size_t total = num_vox_per_pix * imgSize.x * imgSize.y;
  if(!reserved_) {
    allocationList_ = (octlib::key_t* ) calloc(sizeof(octlib::key_t) * total, 1);
    reserved_ = total;
  }
  unsigned int allocated = 0;
  if(std::is_same<FieldType, SDF>::value) {
    allocated  = buildAllocationList(allocationList_, reserved_,
                                     volume_._map_index, T_w_c,
                                     getCameraMatrix(k), depthImage, imgSize,
                                     volume_._resol, voxelsize, 2*mu);
  } else if(std::is_same<FieldType, BFusion>::value) {
    allocated = buildOctantList(allocationList_, reserved_, volume_._map_index,
                                T_w_c, getCameraMatrix(k), depthImage,
                                imgSize, voxelsize,
                                compute_stepsize, step_to_depth, 6*mu);
  }

  volume_._map_index.alloc_update(allocationList_, allocated);

  if(std::is_same<FieldType, SDF>::value) {
//      if (!render_color_){
//        struct sdf_update funct(floatDepth, computationSize, mu, 100);
//        iterators::projective_functor<FieldType, INDEX_STRUCTURE, struct sdf_update>
//          it(volume._map_index, funct, inverse(pose), getCameraMatrix(k),
//             make_int2(computationSize));
//        it.apply();
//      }
//      else{
    struct sdf_update funct(depthImage, rgbImage, imgSize, mu, 100);
    iterators::projective_functor<FieldType, INDEX_STRUCTURE, struct sdf_update>
        it(volume_._map_index, funct, inverse(T_w_c), getCameraMatrix(k),
           make_int2(imgSize));
    it.apply();
//      }

  } else if(std::is_same<FieldType, BFusion>::value) {
    float timestamp = (1.f / 30.f) * frame;
//      if (!render_color_) {
//        struct bfusion_update funct(floatDepth, computationSize, mu, timestamp);
//        iterators::projective_functor<FieldType,
//                                      INDEX_STRUCTURE,
//                                      struct bfusion_update>
//          it(volume._map_index,
//             funct,
//             inverse(pose),
//             getCameraMatrix(k),
//             make_int2(computationSize));
//        it.apply();
//      }
//      else{
    struct bfusion_update funct(depthImage, rgbImage, imgSize, mu, timestamp);
    iterators::projective_functor<FieldType, INDEX_STRUCTURE, struct bfusion_update>
        it(volume_._map_index, funct, inverse(T_w_c), getCameraMatrix(k),
           make_int2(imgSize));
    it.apply();
  }
//    }
}


void Object::integrate_volume_kernel(const float * depthImage,
                                     const float3*rgbImage,
                                     const cv::Mat& mask, const uint2& imgSize,
                                     const Matrix4& T_w_c, const float mu,
                                     const float4& k, const uint frame){

  const float &voxelsize =  this->get_voxel_size();
  int num_vox_per_pix = volume_._size/((VoxelBlock<FieldType>::side) *voxelsize);
  size_t total = num_vox_per_pix * imgSize.x * imgSize.y;
  if(!reserved_) {
    allocationList_ = (octlib::key_t* ) calloc(sizeof(octlib::key_t) * total, 1);
    reserved_ = total;
  }
  unsigned int allocated = 0;
  if(std::is_same<FieldType, SDF>::value) {
    allocated  = buildVolumeAllocationList(allocationList_, reserved_,
                                     volume_._map_index, T_w_c, volume_pose_,
                                     getCameraMatrix(k), depthImage, mask, imgSize,
                                     volume_._resol, voxelsize, 2*mu);
  } else if(std::is_same<FieldType, BFusion>::value) {
    allocated = buildVolumeOctantList(allocationList_, reserved_, volume_._map_index,
                                      T_w_c, volume_pose_, getCameraMatrix(k),
                                      depthImage, mask, imgSize, voxelsize,
                                      compute_stepsize, step_to_depth, 6*mu);
  }

  volume_._map_index.alloc_update(allocationList_, allocated);

  if(std::is_same<FieldType, SDF>::value) {
//      if (!render_color_){
//        struct sdf_update funct(floatDepth, computationSize, mu, 100);
//        iterators::projective_functor<FieldType, INDEX_STRUCTURE, struct sdf_update>
//          it(volume._map_index, funct, inverse(pose), getCameraMatrix(k),
//             make_int2(computationSize));
//        it.apply();
//      }
//      else{
    struct sdf_update funct(depthImage, rgbImage, mask, imgSize, mu, 100);
    iterators::projective_functor<FieldType, INDEX_STRUCTURE, struct sdf_update>
        it(volume_._map_index, funct, inverse(T_w_c)*volume_pose_, getCameraMatrix(k),
           make_int2(imgSize));
    it.apply();
//      }

  } else if(std::is_same<FieldType, BFusion>::value) {
    float timestamp = (1.f / 30.f) * frame;
//      if (!render_color_) {
//        struct bfusion_update funct(floatDepth, computationSize, mu, timestamp);
//        iterators::projective_functor<FieldType,
//                                      INDEX_STRUCTURE,
//                                      struct bfusion_update>
//          it(volume._map_index,
//             funct,
//             inverse(pose),
//             getCameraMatrix(k),
//             make_int2(computationSize));
//        it.apply();
//      }
//      else{
    struct bfusion_update funct(depthImage, rgbImage, mask, imgSize, mu,
        timestamp);
    iterators::projective_functor<FieldType, INDEX_STRUCTURE, struct bfusion_update>
        it(volume_._map_index, funct, inverse(T_w_c)*volume_pose_, getCameraMatrix(k),
           make_int2(imgSize));
    it.apply();
  }
//    }
}

void Object::fuse_semantic_kernel(const SegmentationResult& segmentationResult,
                                  const int instance_id){
  const instance_seg& seg = segmentationResult.pair_instance_seg_.at(instance_id);
  const int class_id = seg.class_id_;

  if ((class_id != 0)) {
    fuseSemanticLabel(seg);
  }

  //semantic fusion
//  if (this->fused_time_ == 0){
//    std::cout<<"first time semanticfusion: input->"<<class_id<<" "
//                                                               "fused->"<<this->class_id_<<std::endl;
//  }
//  else
  if (!seg.rendered_){
//    std::cout<<"semanticfusion: id->"<<instance_id<<" input->"<<class_id
//             <<" fused->"<<this->class_id_<<std::endl;
  }
//  else{
//    std::cout<<"this class not seen on this frame: id->"<<instance_id
//             <<" before->"<<class_id<<" after->"<<this->class_id_<<std::endl;
//  }

}

void Object::fuseSemanticLabel(const instance_seg& input_seg){
  if(label_size_ != input_seg.all_prob_.size()) {
    std::cerr<<"label number is incorrect"<<std::endl;
    exit (EXIT_FAILURE);
  }

   // semanticfusion_ = All_Prob_Vect::Constant(1.4564);
//    std::cout<<input_labels<<std::endl;
//    std::cout<<input_labels.size()<<" "<<input_labels.rows()<<std::endl;
//    std::cout<<semanticfusion_.size()<<" "<<semanticfusion_.rows()<<std::endl;
//    std::cout<<semanticfusion_<<std::endl;
//  semanticfusion_ = input_labels;
//    semanticfusion_ *= fused_time_;
//    semanticfusion_ = semanticfusion_+ input_labels; // /
//        semanticfusion_ /= (fused_time_ + 1);

  const All_Prob_Vect& input_all_prob = input_seg.all_prob_;
  const int recog_times = input_seg.recog_time_;
  semanticfusion_ = (semanticfusion_ * static_cast<float>(fused_time_) +
      input_all_prob * recog_times)
      / static_cast<float>(fused_time_ + recog_times);

  All_Prob_Vect::Index max_id;
  float max_prob = semanticfusion_.maxCoeff(&max_id);

  //get fused class id
  int fused_id = max_id + 1;  // +1 shift due to bg has no probablity
  this->class_id_ = fused_id;
  this->fused_time_ = this->fused_time_ + recog_times;
}


void Object::refine_mask_use_motion(cv::Mat& mask, const bool use_icp,
                                    const bool use_rgb){
  unsigned int y;
/*#pragma omp parallel for \
        shared(mask), private(y)*/
  for (y = 0; y < mask.rows; y++) {
    for (unsigned int x = 0; x < mask.cols; x++) {
      const uint icp_pos = x + y * mask.cols;
      const uint rgb_pos = icp_pos + mask.cols * mask.rows;

      // if within of the band => recognised as objects
      if (mask.at<float>(y,x) >= 0){
        //-4/-5: depth distance or normal is wrong => allocated
        // => now refine/remove high residual areas => don't fuse tsdf
        if (use_icp){
          if ((this->trackresult_[icp_pos].result == -4) ||
              (this->trackresult_[icp_pos].result == -6)) {
//          if ((this->trackresult_[icp_pos].result < -3)) {
            if (absorb_outlier_bg){
//              mask.at<float>(y, x) = 0;
            }
            else{
              mask.at<float>(y, x) = -1;
            }
            continue;
          }
        }

        if (use_rgb){
          if (this->trackresult_[rgb_pos].result < -3) {
            if (absorb_outlier_bg){
//              mask.at<float>(y, x) = 0;
            }
            else{
              mask.at<float>(y, x) = -1;
            }
            continue;
          }
        }
      }


      // if outside of the band => recognised as background
/*      if (mask.at<float>(y,x) < 0){
        //-3: no correspondence in model => not allocated => don't fuse tsdf
//        if (this->trackresult_[pos].result == -3) {
//          mask.at<float>(y,x) = -1;
//          continue;
//        }

        //-4/-5: depth distance or normal is wrong => allocated
        // => now update this area as background
        if (this->trackresult_[rgb_pos].result < -2) {
          mask.at<float>(y,x) = 0;
          continue;
        }
      }*/

    }
  }
}

void Object::set_static_state(const bool state){
  if (static_object.find(this->class_id_) == static_object.end())
    isStatic_ = state;
}

bool Object::is_static() const{
  return this->isStatic_;
}

const std::set<int> Object::static_object({/*50 /* orange,*/ 63 /*tv*/, 69, 70});