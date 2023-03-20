/*
 * SPDX-FileCopyrightText: 2016-2019 Emanuele Vespa
 * SPDX-FileCopyrightText: 2017-2019 Smart Robotics Lab, Imperial College London
 * SPDX-FileCopyrightText: 2017-2019 Binbin Xu
 * SPDX-License-Identifier: BSD-3-Clause
 */

#ifndef BFUSION_MAPPING_HPP
#define BFUSION_MAPPING_HPP

#include <node.hpp>
#include <constant_parameters.h>
#include "bspline_lookup.cc"
#include "../continuous/volume_traits.hpp"
#include "functors/projective_functor.hpp"
#include <opencv2/opencv.hpp>

static inline float interpDepth(const float * depth, const uint2 depthSize,
    const float2 proj) {
  // https://en.wikipedia.org/wiki/Bilinear_interpolation

  // Pixels version
  const float x1 = (floorf(proj.x));
  const float y1 = (floorf(proj.y + 1));
  const float x2 = (floorf(proj.x + 1));
  const float y2 = (floorf(proj.y));

  // Half pixels
  // const float x1 = (float) (int(proj.x - 0.5f)) + 0.5f;
  // const float y1 = (float) (int(proj.y + 0.5f)) + 0.5f;
  // const float x2 = (float) (int(proj.x + 0.5f)) + 0.5f;
  // const float y2 = (float) (int(proj.y - 0.5f)) + 0.5f;

  const float d11 = depth[int(x1) +  depthSize.x*int(y1)];
  const float d12 = depth[int(x1) +  depthSize.x*int(y2)];
  const float d21 = depth[int(x2) +  depthSize.x*int(y1)];
  const float d22 = depth[int(x2) +  depthSize.x*int(y2)];
  
  if( d11 == 0.f || d12 == 0.f || d21 == 0.f || d22 == 0.f ) return 0.f;

  const float f11 = 1.f / d11;
  const float f12 = 1.f / d12;
  const float f21 = 1.f / d21;
  const float f22 = 1.f / d22;
  
  // Filtering version
  const float d =  1.f / 
                    ( (   f11 * (x2 - proj.x) * (y2 - proj.y)
                        + f21 * (proj.x - x1) * (y2 - proj.y)
                        + f12 * (x2 - proj.x) * (proj.y - y1)
                        + f22 * (proj.x - x1) * (proj.y - y1)
                      ) / ((x2 - x1) * (y2 - y1))
                    );

  static const float interp_thresh = 0.05f;
  if (fabs(d - d11) < interp_thresh && fabs(d - d12) < interp_thresh &&
      fabs(d - d21) < interp_thresh && fabs(d - d22) < interp_thresh) 
    return d;
  else 
    return depth[int(proj.x + 0.5f) + depthSize.x*int(proj.y+0.5f)];

  // Non-Filtering version
  // return  1.f / 
  //         ( (   f11 * (x2 - proj.x) * (y2 - proj.y)
  //             + f21 * (proj.x - x1) * (y2 - proj.y)
  //             + f12 * (x2 - proj.x) * (proj.y - y1)
  //             + f22 * (proj.x - x1) * (proj.y - y1)
  //           ) / ((x2 - x1) * (y2 - y1))
  //         );
}

static inline float bspline(float t){
  float value = 0.f;
  if(t >= -3.0f && t <= -1.0f) {
    value = std::pow((3 + t), 3)/48.0f;   
  } else if( t > -1 && t <= 1) {
    value = 0.5f + (t*(3 + t)*(3 - t))/24.f;
  } else if(t > 1 && t <= 3){
    value = 1 - std::pow((3 - t), 3)/48.f;
  } else if(t > 3) {
    value = 1.f;
  }
  return value;
}

static inline float H(const float val){
  const float Q_1 = bspline(val);
  const float Q_2 = bspline(val - 3);
  return Q_1 - Q_2 * 0.5f;
}

static const double const_offset =  0.0000001f;
const float scale_factor = (1.f - (farPlane - nearPlane) * const_offset); 

static inline float const_offset_integral(float t){
  float value = 0.f;
  if (nearPlane <= t && t <= farPlane)
      return (t - nearPlane) * const_offset;
  else if (farPlane < t)
      return (farPlane - nearPlane) * const_offset;
  return value;
}

static inline float __device__ bspline_memoized(float t){
  float value = 0.f;
  constexpr float inverseRange = 1/6.f;
  if(t >= -3.0f && t <= 3.0f) {
    unsigned int idx = ((t + 3.f)*inverseRange)*(bspline_num_samples - 1) + 0.5f;
    return bspline_lookup[idx];
  } 
  else if(t > 3) {
    value = 1.f;
  }
  return value;
}

static inline float HNew(const float val,const  float ){
  const float Q_1 = bspline_memoized(val)    ; // * scale_factor + const_offset_integral(d_xr      );
  const float Q_2 = bspline_memoized(val - 3); // * scale_factor + const_offset_integral(d_xr - 3.f);
  return Q_1 - Q_2 * 0.5f;
}

static inline float updateLogs(const float prior, const float sample){
  // return (prior + clamp(log2(sample / (1.f - sample)), -100, 100));
  return (prior + log2(sample / (1.f - sample)));
}

static inline float applyWindow(const float occupancy, const float , 
    const float delta_t, const float tau){
  float fraction = 1.f / (1.f + (delta_t / tau));
  fraction = std::max(0.5f,fraction);
  return occupancy * fraction;
}

struct bfusion_update {

  template <typename DataHandlerT>
  void operator()(DataHandlerT& handler, const int3&, const float3& pos, 
     const float2& pixel) {

    const uint2 px = make_uint2(pixel.x, pixel.y);
    const float depthSample = depth[px.x + depthSize.x*px.y];
    if (depthSample <=  0|| mask_.at<float>(px.y, px.x) < 0) return;

    const float diff = (pos.z - depthSample)
      * std::sqrt( 1 + sq(pos.x / pos.z) + sq(pos.y / pos.z));
    float sample = HNew(diff/(noiseFactor *sq(pos.z)), pos.z);
    if(sample == 0.5f) return;
    sample = clamp(sample, 0.03f, 0.97f);
    auto data = handler.get();
    const double delta_t = timestamp - data.y;
    data.x = applyWindow(data.x, SURF_BOUNDARY, delta_t, CAPITAL_T);
    data.x = clamp(updateLogs(data.x, sample), BOTTOM_CLAMP, TOP_CLAMP);
    data.y = timestamp;

    //color information
    const float3 rgb_measured = rgb_[px.x + depthSize.x*px.y];
    data.r = (rgb_measured.x + data.r * data.c)/(data.c + 1);
    data.g = (rgb_measured.y + data.g * data.c)/(data.c + 1);
    data.b = (rgb_measured.z + data.b * data.c)/(data.c + 1);
    data.c += 1;

    handler.set(data);
  }

  //grey version
  bfusion_update(const float * d, const uint2 framesize, float n, float t) : 
    depth(d), depthSize(framesize), noiseFactor(n), timestamp(t){};

  //color version
  bfusion_update(const float * d, const float3 *rgb, const uint2 framesize,
                 float n, float t) :
      depth(d), rgb_(rgb), depthSize(framesize), noiseFactor(n), timestamp(t){};

  //color mask version
  bfusion_update(const float * d, const float3 *rgb, const cv::Mat& mask,
                 const uint2 framesize, float n, float t) :
      depth(d), rgb_(rgb), mask_(mask), depthSize(framesize), noiseFactor(n),
      timestamp(t){};

  const float * depth;
  const float3 * rgb_;
  const cv::Mat mask_;
  uint2 depthSize;
  float noiseFactor;
  float timestamp;
};
#endif
