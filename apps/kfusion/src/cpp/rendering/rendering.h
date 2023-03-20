 /*
 * SPDX-FileCopyrightText: 2017-2019 Smart Robotics Lab, Imperial College London
 * SPDX-FileCopyrightText: 2017-2019 Binbin Xu
 * SPDX-License-Identifier: BSD-3-Clause
 */

#ifndef RENDERING_H
#define RENDERING_H

#include <math_utils.h>
#include "timings.h"
#include "../continuous/volume_instance.hpp"
#include <tuple>
#include "../object/object.h"

/* Raycasting implementations */
#include "../bfusion/rendering_impl.hpp"
#include "../kfusion/rendering_impl.hpp"


template<typename T>
void raycastKernel(const Volume<T>& volume, float3* vertex, float3* normal, uint2 inputSize,
                   const Matrix4 view, const float nearPlane, const float farPlane,
                   const float mu, const float step, const float largestep) {
  TICK();
  unsigned int y;
#pragma omp parallel for shared(normal, vertex), private(y)
  for (y = 0; y < inputSize.y; y++)
#pragma simd
      for (unsigned int x = 0; x < inputSize.x; x++) {

        uint2 pos = make_uint2(x, y);
        ray_iterator<typename Volume<T>::field_type> ray(volume._map_index, get_translation(view),
                                                         normalize(rotate(view, make_float3(x, y, 1.f))), nearPlane, farPlane);
        const std::tuple<float, float, float> t = ray.next(); /* Get distance to the first intersected block */
        float t_min = std::get<0>(t);
        const float4 hit = t_min > 0.f ?
                           raycast(volume, pos, view, t_min*volume._size/volume._resol,
                                   farPlane, mu, step, largestep) : make_float4(0.f);
        if(hit.w > 0.0) {
          vertex[pos.x + pos.y * inputSize.x] = make_float3(hit);
          float3 surfNorm = volume.grad(make_float3(hit),
                                        [](const auto& val){ return val.x; });
          if (length(surfNorm) == 0) {
            //normal[pos] = normalize(surfNorm); // APN added
            normal[pos.x + pos.y * inputSize.x].x = INVALID;
          } else {
            normal[pos.x + pos.y * inputSize.x] = normalize(surfNorm);
          }
        } else {
          //std::cerr<< "RAYCAST MISS "<<  pos.x << " " << pos.y <<"  " << hit.w <<"\n";
          vertex[pos.x + pos.y * inputSize.x] = make_float3(0);
          normal[pos.x + pos.y * inputSize.x] = make_float3(INVALID, 0, 0);
        }
      }
  TOCK("raycastKernel", inputSize.x * inputSize.y);
}


template <typename T>
void renderVolumeKernel(const Volume<T>& volume, uchar4* out, const uint2 depthSize, const Matrix4 view,
                        const float nearPlane, const float farPlane, const float mu,
                        const float step, const float largestep, const float3 light,
                        const float3 ambient, bool render, bool render_color,
                        const float3 *vertex,
                        const float3 * normal) {
  TICK();
  unsigned int y;
#pragma omp parallel for shared(out), private(y)
  for (y = 0; y < depthSize.y; y++) {
    for (unsigned int x = 0; x < depthSize.x; x++) {
      const uint2 pos = make_uint2(x, y);

      float4 hit;
      float3 test, surfNorm;
      if(render) {
        ray_iterator<typename Volume<T>::field_type> ray(volume._map_index,
                                                         get_translation(view),
                                                         normalize(rotate(view, make_float3(x, y, 1.f))),
                                                         nearPlane,
                                                         farPlane);
        const float t_min = std::get<0>(ray.next()); /* Get distance to the first intersected block */
        hit = t_min > 0.f ?
              raycast(volume, pos, view, t_min*volume._size/volume._resol,
                      farPlane, mu, step, largestep) : make_float4(0.f);
        if (hit.w > 0) {
          test = make_float3(hit);
          surfNorm = volume.grad(test, [](const auto& val){ return val.x; });
        } else {
          out[x + depthSize.x*y] = make_uchar4(0, 0, 0, 0); // The forth value is a padding to align memory
          continue;
        }
      } else {
        test = vertex[x + depthSize.x*y];
        surfNorm = normal[x + depthSize.x*y];
      }

      if (length(surfNorm) > 0) {
        const float3 diff = (std::is_same<T, SDF>::value ?
                             normalize(light - test) : normalize(test - light));
        const float dir = fmaxf(dot(normalize(surfNorm), diff),
                                0.f);
        float3 col;
        if (render_color){
          const float interpolated_r = volume.interp(test, [](const auto&
          val){ return val.r;});
          const float interpolated_g = volume.interp(test, [](const auto&
          val){ return val.g;});
          const float interpolated_b = volume.interp(test, [](const auto&
          val){ return val.b;});
          const float3 rgb = make_float3(interpolated_r, interpolated_g,
                                         interpolated_b);
          col  = clamp(make_float3(dir) * rgb + ambient, 0.f, 1.f) * 255;
        } else{
          col = clamp(make_float3(dir) + ambient, 0.f, 1.f) * 255;
        }
        out[x + depthSize.x*y] = make_uchar4(col.x, col.y, col.z, 0); // The forth value is a padding to align memory
      }
    }
  }
  TOCK("renderVolumeKernel", depthSize.x * depthSize.y);
}

void raycastObjectList(ObjectList& objectlist, float3* vertex, float3*normal,
                       cv::Mat& labelImg, std::set<int>& objects_in_view,
                       const Matrix4& T_w_c, const float4& k, const uint2& inputSize,
                       const float nearPlane, const float farPlane,
                       const float mu,  const bool has_integrated);

void render_RGBD_TrackKernel(uchar4* out, const TrackData* icp_data,
                             const TrackData* rgb_data, uint2 outSize);


void volume2MaskKernel(const ObjectList& objectlist, cv::Mat& labelImg,
                       const Matrix4&T_w_c, const float4& k, const uint2& inputSize,
                       const float nearPlane, const float farPlane,
                       const float mu);

void renderNormalKernel(uchar3* out, const float3* normal, uint2 normalSize);

void renderDepthKernel(uchar4* out, float * depth, uint2 depthSize,
                       const float nearPlane, const float farPlane);

void renderTrackKernel(uchar4* out, const TrackData* data, uint2 outSize);

void renderRGBTrackKernel(uchar4* out, const TrackData* data, uint2 outSize);

void renderVolume_many_Kernel(const ObjectList& objectlist, uchar4* out,
                              const uint2 depthSize, const Matrix4& T_w_c,
                              const float4& k, const float nearPlane,
                              const float farPlane, const float mu,
                              const float largestep, const float3 ambient,
                              bool doraycast, bool renderColor,
                              const float3 *vertex, const float3 * normal,
                              const cv::Mat& labelImg,
                              const std::vector<uchar4> &colors);

void renderIntensityKernel(uchar4* out, float * intensity, uint2 framesize);

std::vector<uchar4> random_color(int class_nums);

void renderInstanceMaskKernel(uchar4* out, uint2 framesize,
                              const SegmentationResult& mask,
                              const std::vector<uchar4> &colors);

void renderMaskWithImageKernel(uchar4* out, uint2 framesize,
                               const float3* input,
                               const SegmentationResult& mask,
                               const std::vector<uchar4> &colors);

void renderMaskMotionWithImageKernel(uchar4* out, uint2 framesize,
                                     const float3* input,
                                     const SegmentationResult& mask,
                                     const ObjectList& objectlist,
                                     const std::vector<uchar4> &colors);

void renderClassMaskKernel(uchar4* out, uint2 framesize,
                           const SegmentationResult& mask,
                           const std::vector<uchar4> &colors);

void renderMaskKernel(uchar4* out, float * intensity, uint2 framesize,
                      cv::Mat labelImg, std::vector<uchar4> colors);

void renderIntensityKernel(uchar4* out, float * intensity, uint2 framesize,
                           std::vector<cv::Mat> idx_pixel,
                           std::vector<int> class_id,
                           std::vector<uchar4> colors);

void check_static_state(cv::Mat& object_inlier_mask,
                        const cv::Mat& object_mask,
                        const TrackData* camera_tracking_result,
                        const uint2 framesize,
                        const bool use_icp,
                        const bool use_rgb);

void opengl2opencv_kernel(cv::Mat& renderImg, const uchar4* data, const
uint2 outSize);

void opengl2opencv(const uchar4* data, const uint2 outSize, const uint frame,
                   const std::string filename);

#endif //RENDERING_H
