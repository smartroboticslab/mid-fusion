/*

Copyright 2019 Binbin Xu, Imperial College London 
Copyright 2016 Emanuele Vespa, Imperial College London 

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation
and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
may be used to endorse or promote products derived from this software without
specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE. 

*/


#ifndef MATH_UTILS_H
#define MATH_UTILS_H

#include <iostream>
#include <cmath>
#ifndef CUDA
#include "vector_types.h"
#else
#include <cuda_runtime.h>
#endif
#include "cutil_math.h"


typedef struct sMatrix4 {
	float4 data[4];
} Matrix4;

inline __host__ __device__ float3 get_translation(const Matrix4& view) {
	return make_float3(view.data[0].w, view.data[1].w, view.data[2].w);
}

////////////////////////////////////////////////////////////////////////////////
// ceilf - missing from cutil_math.h
////////////////////////////////////////////////////////////////////////////////

inline __host__     __device__ float2 ceilf(float2 v) {
	return make_float2(ceilf(v.x), ceilf(v.y));
}
inline __host__     __device__ float3 ceilf(float3 v) {
	return make_float3(ceilf(v.x), ceilf(v.y), ceilf(v.z));
}
inline __host__     __device__ float4 ceilf(float4 v) {
	return make_float4(ceilf(v.x), ceilf(v.y), ceilf(v.z), ceilf(v.w));
}

inline __host__ __device__ bool operator==(const float3 a, float b){
  return((a.x == b) && (a.y == b) && (a.z == b));
}

inline __host__ __device__ bool in(const unsigned int value, const unsigned int lower, 
               const unsigned int upper){
  return value >= lower && value <= upper;
}

inline __host__ __device__ bool in(const int value, const int lower, 
               const int upper){
  return value >= lower && value <= upper;
}

inline __host__     __device__ uchar3 operator*(const uchar3 a, float v) {
	return make_uchar3(a.x * v, a.y * v, a.z * v);
}

inline float4 operator*(const Matrix4 & M, const float4 & v) {
	return make_float4(dot(M.data[0], v), dot(M.data[1], v), dot(M.data[2], v),
			dot(M.data[3], v));
}

inline float sq(float r) {
	return r * r;
}

inline Matrix4 outer(const float4& a, const float4& b){
  Matrix4 mat;
  mat.data[0] = make_float4(a.x*b.x, a.x*b.y, a.x*b.z, a.x*b.w);
  mat.data[1] = make_float4(a.y*b.x, a.y*b.y, a.y*b.z, a.y*b.w);
  mat.data[2] = make_float4(a.z*b.x, a.z*b.y, a.z*b.z, a.z*b.w);
  mat.data[3] = make_float4(a.w*b.x, a.w*b.y, a.w*b.z, a.w*b.w);
  return mat;
}

inline __host__      __device__ float3 operator*(const Matrix4 & M,
		const float3 & v) {
	return make_float3(dot(make_float3(M.data[0]), v) + M.data[0].w,
			dot(make_float3(M.data[1]), v) + M.data[1].w,
			dot(make_float3(M.data[2]), v) + M.data[2].w);
}

inline float3 rotate(const Matrix4 & M, const float3 & v) {
	return make_float3(dot(make_float3(M.data[0]), v),
			dot(make_float3(M.data[1]), v), dot(make_float3(M.data[2]), v));
}

inline float3 rotate(const float3 & v, const Matrix4 & M) {
  return make_float3(dot(make_float3(M.data[0].x, M.data[1].x, M.data[2].x), v),
                     dot(make_float3(M.data[0].y, M.data[1].y, M.data[2].y), v),
                     dot(make_float3(M.data[0].z, M.data[1].z, M.data[2].z), v));
}

inline Matrix4 transpose(const Matrix4& M){
  Matrix4 transposed;
  transposed.data[0] = make_float4(M.data[0].x, M.data[1].x, M.data[2].x, M.data[3].x);
  transposed.data[1] = make_float4(M.data[0].y, M.data[1].y, M.data[2].y, M
      .data[3].y);
  transposed.data[2] = make_float4(M.data[0].z, M.data[1].z, M.data[2].z, M
      .data[3].z);
  transposed.data[3] = make_float4(M.data[0].w, M.data[1].w, M.data[2].w, M
      .data[3].w);
  return  transposed;
}

// Converting quaternion and trans to SE3 matrix 
// Following the implementation provided in TUM scripts.
inline Matrix4 toMatrix4(float4 quat, const float3& trans) {
  const float n = dot(quat, quat);
  quat = quat*(sqrtf(2.0/n));
  Matrix4 mat = outer(quat, quat);
  Matrix4 se3_mat;
  se3_mat.data[0] = make_float4(1.0-mat.data[1].y - mat.data[2].z, mat.data[0].y - mat.data[2].w,     mat.data[0].z + mat.data[1].w, trans.x);
  se3_mat.data[1] = make_float4(mat.data[0].y + mat.data[2].w, 1.0-mat.data[0].x - mat.data[2].z,     mat.data[1].z - mat.data[0].w, trans.y);
  se3_mat.data[2] = make_float4(mat.data[0].z - mat.data[1].w,     mat.data[1].z + mat.data[0].w, 1.0-mat.data[0].x - mat.data[1].y, trans.z);
  se3_mat.data[3] = make_float4(0.0, 0.0, 0.0, 1.0);
  return se3_mat;
}

template <typename T>
inline T bilinear_interp(const T* frame, const uint2 framesize, 
    float2 pos) {

  const int2 base = make_int2(floorf(pos));
  const float2 factor = fracf(pos);

  T vals[4];
  vals[0] = frame[base.x + base.y*framesize.x];
  vals[1] = frame[base.x + 1 + base.y*framesize.x];
  vals[2] = frame[base.x + (base.y + 1)*framesize.x];
  vals[3] = frame[base.x + 1 + (base.y + 1)*framesize.x];

  return vals[0]*(1.f - factor.x)*(1.f - factor.y)
       + vals[1]*(1.f - factor.y)*factor.x
       + vals[2]*(1.f - factor.x)*factor.y
       + vals[3]*factor.x*factor.y;
}

constexpr int log2_const(int n){
  return (n < 2 ? 0 : 1 + log2_const(n/2));
}


static inline void compareGTPose(const Matrix4 estimated, const Matrix4 GT, float& trans_diff){
  const float3 estimated_trans = get_translation(estimated);
  const float3 GT_trans = get_translation(GT);
  trans_diff = length(estimated_trans - GT_trans);
}

static inline std::ostream& operator<<(std::ostream& os, const uint3& val) {
  os << "(" << val.x << ", " << val.y << ", " << val.z << ")";
  return os;
}

#endif
