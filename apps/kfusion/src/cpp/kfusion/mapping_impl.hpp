/*
 * SPDX-FileCopyrightText: 2016-2019 Emanuele Vespa
 * SPDX-FileCopyrightText: 2017-2019 Smart Robotics Lab, Imperial College London
 * SPDX-FileCopyrightText: 2017-2019 Binbin Xu
 * SPDX-License-Identifier: BSD-3-Clause
 */
 
 
#ifndef KFUSION_MAPPING_HPP
#define KFUSION_MAPPING_HPP
#include <node.hpp>
#include "../continuous/volume_traits.hpp"
#include <opencv2/opencv.hpp>

struct sdf_update {

  template <typename DataHandlerT>
  void operator()(DataHandlerT& handler, const int3&, const float3& pos, 
     const float2& pixel) {

    const uint2 px = make_uint2(pixel.x, pixel.y);
    const float depthSample = depth[px.x + depthSize.x*px.y];
    if (depthSample <=  0 || mask_.at<float>(px.y, px.x) < 0) return;
    const float diff = (depthSample - pos.z) 
      * std::sqrt( 1 + sq(pos.x / pos.z) + sq(pos.y / pos.z));
    if (diff > -mu) {
      const float sdf = fminf(1.f, diff / mu);
      auto data = handler.get();
//      tsdf
      data.x = clamp((data.y * data.x + sdf) / (data.y + 1), -1.f,
          1.f);
//      weight
      data.y = fminf(data.y + 1, maxweight);

      //color information
      const float3 rgb_measured = rgb_[px.x + depthSize.x*px.y];
      data.r = (rgb_measured.x + data.r * data.c)/(data.c + 1);
      data.g = (rgb_measured.y + data.g * data.c)/(data.c + 1);
      data.b = (rgb_measured.z + data.b * data.c)/(data.c + 1);
      data.c += 1;

      //foreground probability
      const float fg_measured = mask_.at<float>(px.y, px.x);
      data.fg = (fg_measured + data.fg * data.w)/(data.w + 1);
      data.w ++;

      handler.set(data);
    }
  } 

  //grey version
  sdf_update(const float * d, const uint2 framesize, float m, int mw) : 
    depth(d), rgb_(nullptr), depthSize(framesize), mu(m), maxweight(mw){};

  //color version
  sdf_update(const float *d, const float3 *rgb, const uint2 framesize, float m,
             int mw) :
    depth(d), rgb_(rgb), depthSize(framesize), mu(m), maxweight(mw){};

  //mask color version
  sdf_update(const float *d, const float3 *rgb, const cv::Mat& mask,
             const uint2 framesize, float m, int mw) :
      depth(d), rgb_(rgb), mask_(mask), depthSize(framesize), mu(m),
      maxweight(mw){};

  const float * depth;
  const float3 * rgb_;
  const cv::Mat mask_;
  uint2 depthSize;
  float mu;
  int maxweight;
};

#endif
