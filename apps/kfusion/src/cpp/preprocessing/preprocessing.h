/*
 * SPDX-FileCopyrightText: 2016-2019 Emanuele Vespa
 * SPDX-FileCopyrightText: 2017-2019 Smart Robotics Lab, Imperial College London
 * SPDX-FileCopyrightText: 2017-2019 Binbin Xu
 * SPDX-License-Identifier: BSD-3-Clause
 */
 

#ifndef PREPROCESSING_H
#define PREPROCESSING_H

#include "timings.h"
#include <math_utils.h>
#include <functional>
#include "commons.h"
#include "../continuous/volume_instance.hpp"


void bilateralFilterKernel(float* out, const float* in, uint2 size,
                           const float * gaussian, float e_d, int r);

void depth2vertexKernel(float3* vertex, const float * depth, uint2 imageSize,
                               const Matrix4 invK);

void rgb2intensity(float * grey_out, float3 *rgb_out, uint2 outSize, const uchar3 * in,
                   uint2 inSize);

void rgb2intensity(float * out, uint2 outSize, const uchar3 * in,
                          uint2 inSize);

void mm2metersKernel(float * out, uint2 outSize, const ushort * in,
                            uint2 inSize);

void halfSampleRobustImageKernel(float* out, const float* in, uint2 inSize,
                                        const float e_d, const int r);

void gradientsKernel(float * gradx, float * grady, const float * in, uint2 size);

template <typename FieldType, bool NegY>
void vertex2normalKernel(float3 * out, const float3 * in, uint2 imageSize) {
  TICK();
  unsigned int x, y;
#pragma omp parallel for \
        shared(out), private(x,y)
  for (y = 0; y < imageSize.y; y++) {
    for (x = 0; x < imageSize.x; x++) {
      const uint2 pleft = make_uint2(max(int(x) - 1, 0), y);
      const uint2 pright = make_uint2(min(x + 1, (int) imageSize.x - 1),
                                      y);

      // Swapped to match the left-handed coordinate system of ICL-NUIM
      uint2 pup, pdown;
      if(NegY) {
        pup = make_uint2(x, max(int(y) - 1, 0));
        pdown = make_uint2(x, min(y + 1, ((int) imageSize.y) - 1));
      } else {
        pdown = make_uint2(x, max(int(y) - 1, 0));
        pup = make_uint2(x, min(y + 1, ((int) imageSize.y) - 1));
      }

      const float3 left = in[pleft.x + imageSize.x * pleft.y];
      const float3 right = in[pright.x + imageSize.x * pright.y];
      const float3 up = in[pup.x + imageSize.x * pup.y];
      const float3 down = in[pdown.x + imageSize.x * pdown.y];

      if (left.z == 0 || right.z == 0 || up.z == 0 || down.z == 0) {
        out[x + y * imageSize.x].x = INVALID;
        continue;
      }
      const float3 dxv = right - left;
      const float3 dyv = up - down;
      if(std::is_same<FieldType, SDF>::value) {
        out[x + y * imageSize.x] =  normalize(cross(dyv, dxv)); // switched dx and dy to get factor -1
      }
      else if(std::is_same<FieldType, BFusion>::value) {
        out[x + y * imageSize.x] =  normalize(cross(dxv, dyv)); // switched dx and dy to get factor -1
      }
    }
  }
  TOCK("vertex2normalKernel", imageSize.x * imageSize.y);
}

void vertex2depth (float *rendered_depth, const float3 *vertex, const uint2 inputSize, const Matrix4& pose);

Eigen::Quaternionf getQuaternion(const Matrix4& Trans);

#endif //PREPROCESSING_H
