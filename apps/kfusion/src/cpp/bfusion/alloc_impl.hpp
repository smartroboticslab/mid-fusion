/*
 * SPDX-FileCopyrightText: 2016-2019 Emanuele Vespa
 * SPDX-FileCopyrightText: 2017-2019 Smart Robotics Lab, Imperial College London
 * SPDX-FileCopyrightText: 2017-2019 Binbin Xu
 * SPDX-License-Identifier: BSD-3-Clause
 */

#ifndef BFUSION_ALLOC_H
#define BFUSION_ALLOC_H
#include "math_utils.h"
#include <opencv2/opencv.hpp>

/* Compute step size based on distance travelled along the ray */ 
static inline float compute_stepsize(const float dist_travelled, const float hf_band,
    const float voxelSize) {
  float new_step;
  float half = hf_band * 0.5f;
  if(dist_travelled < hf_band) new_step = voxelSize;
  else if(dist_travelled < hf_band + half) new_step = 10.f * voxelSize; 
  else new_step = 30.f * voxelSize;
  return new_step;
}

/* Compute octree level given a step size */ 
static inline int step_to_depth(const float step, const int max_depth, 
    const float voxelsize) {
  return static_cast<int>(floorf(std::log2f(voxelsize/step)) + max_depth);
}

template <typename FieldType, 
          template <typename> class OctreeT, typename HashType,
          typename StepF, typename DepthF>
size_t buildOctantList(HashType* allocationList, size_t reserved,
    OctreeT<FieldType>& map_index, const Matrix4 &pose, 
    const Matrix4& K, const float *depthmap, const uint2 &imageSize, 
    const float voxelSize, StepF compute_stepsize, DepthF step_to_depth,
    const float band) {

  const float inverseVoxelSize = 1.f/voxelSize;
  Matrix4 invK = inverse(K);
  const Matrix4 kPose = pose * invK;
  const int size = map_index.size();
  const int max_depth = log2(size);
  const int leaves_depth = OctreeT<FieldType>::block_depth;

#ifdef _OPENMP
  std::atomic<unsigned int> voxelCount;
  std::atomic<unsigned int> leavesCount;
#else
  unsigned int voxelCount;
#endif

  unsigned int x, y;
  const float3 camera = get_translation(pose);
  voxelCount = 0;
#pragma omp parallel for \
  private(y)
  for (y = 0; y < imageSize.y; y++) {
    for (x = 0; x < imageSize.x; x++) {
      if(depthmap[x + y*imageSize.x] == 0)
        continue;
      int tree_depth = max_depth; 
      float stepsize = voxelSize;
      const float depth = depthmap[x + y*imageSize.x];
      float3 worldVertex = (kPose * make_float3((x + 0.5f) * depth, 
            (y + 0.5f) * depth, depth));

      float3 direction = normalize(camera - worldVertex);
      const float3 origin = worldVertex - (band * 0.5f) * direction;
      const float dist = length(camera - origin); 
      float3 step = direction*stepsize;

      float3 voxelPos = origin;
      float travelled = 0.f;
      for(; travelled < dist; travelled += stepsize){

        float3 voxelScaled = floorf(voxelPos * inverseVoxelSize);
        if((voxelScaled.x < size) && (voxelScaled.y < size) &&
           (voxelScaled.z < size) && (voxelScaled.x >= 0) &&
           (voxelScaled.y >= 0) && (voxelScaled.z >= 0)){
          const int3 voxel = make_int3(voxelScaled);
          auto node_ptr = map_index.fetch_octant(voxel.x, voxel.y, voxel.z, 
              tree_depth);
          if(!node_ptr){
            HashType k = map_index.hash(voxel.x, voxel.y, voxel.z, tree_depth);
            unsigned int idx = ++(voxelCount);
            if(idx < reserved) {
              allocationList[idx] = k;
            }
          } else if(tree_depth >= leaves_depth) { 
            static_cast<VoxelBlock<FieldType>*>(node_ptr)->active(true);
          }
        }
        stepsize = compute_stepsize(travelled, band, voxelSize);  
        // int last_depth = tree_depth;
        tree_depth = step_to_depth(stepsize, max_depth, voxelSize);
        // if(tree_depth != last_depth) {
        //   std::cout << "Break Here!" << std::endl;
        // }
        
        step = direction*stepsize;
        voxelPos +=step;
      }
    }
  }
  return (size_t) voxelCount >= reserved ? reserved : (size_t) voxelCount;
}

template <typename FieldType,
    template <typename> class OctreeT, typename HashType,
    typename StepF, typename DepthF>
size_t buildVolumeOctantList(HashType* allocationList, size_t reserved,
                             OctreeT<FieldType>& map_index, const Matrix4 &T_w_c,
                             const Matrix4 &T_w_o, const Matrix4& K,
                             const float *depthmap, const cv::Mat& mask,
                             const uint2 &imageSize, const float voxelSize,
                             StepF compute_stepsize, DepthF step_to_depth,
                             const float band) {

  const float inverseVoxelSize = 1.f/voxelSize;
  Matrix4 invK = inverse(K);
  const Matrix4 kPose = T_w_c * invK;
  const int size = map_index.size();
  const int max_depth = log2(size);
  const int leaves_depth = OctreeT<FieldType>::block_depth;

#ifdef _OPENMP
  std::atomic<unsigned int> voxelCount;
  std::atomic<unsigned int> leavesCount;
#else
  unsigned int voxelCount;
#endif

  unsigned int x, y;
  const float3 w_camera_center = get_translation(T_w_c);
  const float3 o_camera_center = inverse(T_w_o) * w_camera_center;
  voxelCount = 0;
#pragma omp parallel for \
  private(y)
  for (y = 0; y < imageSize.y; y++) {
    for (x = 0; x < imageSize.x; x++) {
      if(depthmap[x + y*imageSize.x] == 0)
        continue;
      int tree_depth = max_depth;
      float stepsize = voxelSize;
      const float depth = depthmap[x + y*imageSize.x];
      float3 w_vertex = (kPose * make_float3((x + 0.5f) * depth,
                                             (y + 0.5f) * depth, depth));
      float3 o_vertex = inverse(T_w_o) * w_vertex;

      float3 direction = normalize(o_camera_center - o_vertex);
      const float3 origin = o_vertex - (band * 0.5f) * direction;
      const float dist = length(o_camera_center - origin);
      float3 step = direction*stepsize;

      float3 voxelPos = origin;
      float travelled = 0.f;
      for(; travelled < dist; travelled += stepsize){

        float3 voxelScaled = floorf(voxelPos * inverseVoxelSize);
        if((voxelScaled.x < size) && (voxelScaled.y < size) &&
            (voxelScaled.z < size) && (voxelScaled.x >= 0) &&
            (voxelScaled.y >= 0) && (voxelScaled.z >= 0)){
          const int3 voxel = make_int3(voxelScaled);
          auto node_ptr = map_index.fetch_octant(voxel.x, voxel.y, voxel.z,
                                                 tree_depth);
          if(!node_ptr){
            HashType k = map_index.hash(voxel.x, voxel.y, voxel.z, tree_depth);
            unsigned int idx = ++(voxelCount);
            if(idx < reserved) {
              allocationList[idx] = k;
            }
          } else if(tree_depth >= leaves_depth) {
            static_cast<VoxelBlock<FieldType>*>(node_ptr)->active(true);
          }
        }
        stepsize = compute_stepsize(travelled, band, voxelSize);
        // int last_depth = tree_depth;
        tree_depth = step_to_depth(stepsize, max_depth, voxelSize);
        // if(tree_depth != last_depth) {
        //   std::cout << "Break Here!" << std::endl;
        // }

        step = direction*stepsize;
        voxelPos +=step;
      }
    }
  }
  return (size_t) voxelCount >= reserved ? reserved : (size_t) voxelCount;
}
#endif
