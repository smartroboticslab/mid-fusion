/*
 * SPDX-FileCopyrightText: 2016-2019 Emanuele Vespa
 * SPDX-FileCopyrightText: 2017-2019 Smart Robotics Lab, Imperial College London
 * SPDX-FileCopyrightText: 2017-2019 Binbin Xu
 * SPDX-License-Identifier: BSD-3-Clause
 */
 

#ifndef KFUSION_RENDERING_IMPL_H
#define kFUSION_RENDERING_IMPL_H

#include <math_utils.h>
#include <type_traits>

inline float4 raycast(const Volume<SDF>& volume, const uint2 pos,
                      const Matrix4 view, const float nearPlane,
                      const float farPlane, const float mu,
                      const float step, const float largestep) {

  const float3 origin = get_translation(view);
  const float3 direction = normalize(rotate(view, make_float3(pos.x, pos.y, 1.f)));

  // intersect ray with a box
  // http://www.siggraph.org/education/materials/HyperGraph/raytrace/rtinter3.htm
  // compute intersection of ray with all six bbox planes
  const float3 invR = make_float3(1.0f) / direction;
  const float3 tbot = -1 * invR * origin;
  const float3 ttop = invR * (volume._size - origin);

  // re-order intersections to find smallest and largest on each axis
  const float3 tmin = fminf(ttop, tbot);
  const float3 tmax = fmaxf(ttop, tbot);

  // find the largest tmin and the smallest tmax
  const float largest_tmin = fmaxf(fmaxf(tmin.x, tmin.y),
      fmaxf(tmin.x, tmin.z));
  const float smallest_tmax = fminf(fminf(tmax.x, tmax.y),
      fminf(tmax.x, tmax.z));

  // check against near and far plane
  const float tnear = fmaxf(largest_tmin, nearPlane);
  const float tfar = fminf(smallest_tmax, farPlane);
  auto select_depth = [](const auto& val){ return val.x; };

  if (tnear < tfar) {
    // first walk with largesteps until we found a hit
    float t = tnear;
    float stepsize = largestep;
    float3 position = origin + direction * t;
    float f_t = volume.interp(position, select_depth);
    float f_tt = 0;
    if (f_t > 0) { // ups, if we were already in it, then don't render anything here
      for (; t < tfar; t += stepsize) {
        Volume<SDF>::compute_type data = volume.get(position);
        if(data.y < 0){
          stepsize = largestep;
          position += stepsize*direction;
          continue;
        }
        f_tt = data.x;
        if(f_tt <= 0.1 && f_tt >= -0.5f){
          f_tt = volume.interp(position, select_depth);
        }
        if (f_tt < 0)                  // got it, jump out of inner loop
          break;
        stepsize = fmaxf(f_tt * mu, step);
        position += stepsize*direction;
        //stepsize = step;
        f_t = f_tt;
      }
      if (f_tt < 0) {           // got it, calculate accurate intersection
        t = t + stepsize * f_tt / (f_t - f_tt);
        return make_float4(origin + direction * t, t);
      }
    }
  }
  return make_float4(0);
}

inline float4 raycast(const Volume<SDF>& volume, const float3 origin,
                      const float3 direction, const float nearPlane,
                      const float farPlane, const float mu,
                      const float step, const float largestep) {

  // intersect ray with a box
  // http://www.siggraph.org/education/materials/HyperGraph/raytrace/rtinter3.htm
  // compute intersection of ray with all six bbox planes
  const float3 invR = make_float3(1.0f) / direction;
  const float3 tbot = -1 * invR * origin;
  const float3 ttop = invR * (volume._size - origin);

  // re-order intersections to find smallest and largest on each axis
  const float3 tmin = fminf(ttop, tbot);
  const float3 tmax = fmaxf(ttop, tbot);

  // find the largest tmin and the smallest tmax
  const float largest_tmin = fmaxf(fmaxf(tmin.x, tmin.y),
                                   fmaxf(tmin.x, tmin.z));
  const float smallest_tmax = fminf(fminf(tmax.x, tmax.y),
                                    fminf(tmax.x, tmax.z));

  // check against near and far plane
  const float tnear = fmaxf(largest_tmin, nearPlane);
  const float tfar = fminf(smallest_tmax, farPlane);
  auto select_depth = [](const auto& val){ return val.x; };

  if (tnear < tfar) {
    // first walk with largesteps until we found a hit
    float t = tnear;
    float stepsize = largestep;
    float3 position = origin + direction * t;
    float f_t = volume.interp(position, select_depth);
    float f_tt = 0;
    if (f_t > 0) { // ups, if we were already in it, then don't render anything here
      for (; t < tfar; t += stepsize) {
        Volume<SDF>::compute_type data = volume.get(position);
        if(data.y < 0){
          stepsize = largestep;
          position += stepsize*direction;
          continue;
        }
        f_tt = data.x;
        if(f_tt <= 0.1 && f_tt >= -0.5f){
          f_tt = volume.interp(position, select_depth);
        }
        if (f_tt < 0)                  // got it, jump out of inner loop
          break;
        stepsize = fmaxf(f_tt * mu, step);
        position += stepsize*direction;
        //stepsize = step;
        f_t = f_tt;
      }
      if (f_tt < 0) {           // got it, calculate accurate intersection
        t = t + stepsize * f_tt / (f_t - f_tt);
        return make_float4(origin + direction * t, t);
      }
    }
  }
  return make_float4(0);
}


#endif //KFUSION_RENDERING_IMPL_H
