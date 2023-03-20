/*
 * SPDX-FileCopyrightText: 2016-2019 Emanuele Vespa
 * SPDX-FileCopyrightText: 2017-2019 Smart Robotics Lab, Imperial College London
 * SPDX-FileCopyrightText: 2017-2019 Binbin Xu
 * SPDX-License-Identifier: BSD-3-Clause
 */
 
 
#ifndef VOLUME_H
#define VOLUME_H

// Data types definitions
#include "voxel_traits.hpp"

/******************************************************************************
 *
 * KFusion Truncated Signed Distance Function voxel traits
 *
****************************************************************************/

class SDF {};
template<>
struct voxel_traits<SDF> {
//  typedef float2 ComputeType;
//  typedef float2 StoredType;
  typedef struct ComputeType {
    float x;  //tsdf
    float y;  //weight
    float r;  //color information: R
    float g;  //color information: G
    float b;  //color information: B
    int c;   //color votes
    double fg; //foreground probability
    int w; //foreground witness
  } ComputeType;
  typedef ComputeType StoredType;
//  static inline ComputeType empty(){ return make_float2(1.f, -1.f); }
  static inline ComputeType empty(){ return {1.f, -1.f, 0.f, 0.f, 0.f, 0, 0.f, 0}; }
//  static inline StoredType initValue(){ return make_float2(1.f, 0.f); }
  static inline StoredType initValue(){ return {1.f, 0.f, 0.f, 0.f, 0.f, 0, 0.f, 0}; }
  static inline ComputeType translate(const StoredType value){
//    return make_float2(value.x, value.y);
    return value;
  }
};

/******************************************************************************
 *
 * Bayesian Fusion voxel traits and algorithm specificic defines
 *
****************************************************************************/

class BFusion {};
template<>
struct voxel_traits<BFusion> {
//  typedef struct ComputeType {
//    float x;
//    double y;
//  } ComputeType;
  typedef struct ComputeType {
    float x;
    double y;
    float r;  //color information: R
    float g;  //color information: G
    float b;  //color information: B
    int c;   //color votes
    double fg; //foreground probability
    int w; //foreground witness
  } ComputeType;
  typedef ComputeType StoredType;
//  static inline ComputeType empty(){ return {0.f, 0.f}; }
//  static inline StoredType initValue(){ return {0.f, 0.f}; }
  static inline ComputeType empty(){ return {0.f, 0.f, 0.f, 0.f, 0.f, 0, 0.f, 0}; }
  static inline StoredType initValue(){ return {0.f, 0.f, 0.f, 0.f, 0.f, 0, 0.f, 0}; }
  static inline StoredType translate(const ComputeType value) {
     return value;
  }
};

// Windowing parameters
#define DELTA_T   1.f
#define CAPITAL_T 4.f

#define INTERP_THRESH 0.05f
#define SURF_BOUNDARY 0.f
#define TOP_CLAMP     1000.f
#define BOTTOM_CLAMP  (-TOP_CLAMP)

#endif
