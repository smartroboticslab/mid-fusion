/*
 * SPDX-FileCopyrightText: 2016-2019 Emanuele Vespa
 * SPDX-FileCopyrightText: 2017-2019 Smart Robotics Lab, Imperial College London
 * SPDX-FileCopyrightText: 2017-2019 Binbin Xu
 * SPDX-License-Identifier: BSD-3-Clause
 */
 
 
#include "preprocessing.h"

void bilateralFilterKernel(float* out, const float* in, uint2 size,
		const float * gaussian, float e_d, int r) {
	TICK()
		uint y;
		float e_d_squared_2 = e_d * e_d * 2;
#pragma omp parallel for \
	    shared(out),private(y)   
		for (y = 0; y < size.y; y++) {
			for (uint x = 0; x < size.x; x++) {
				uint pos = x + y * size.x;
				if (in[pos] == 0) {
					out[pos] = 0;
					continue;
				}

				float sum = 0.0f;
				float t = 0.0f;

				const float center = in[pos];

				for (int i = -r; i <= r; ++i) {
					for (int j = -r; j <= r; ++j) {
						uint2 curPos = make_uint2(clamp(x + i, 0u, size.x - 1),
								clamp(y + j, 0u, size.y - 1));
						const float curPix = in[curPos.x + curPos.y * size.x];
						if (curPix > 0) {
							const float mod = sq(curPix - center);
							const float factor = gaussian[i + r]
									* gaussian[j + r]
									* expf(-mod / e_d_squared_2);
							t += factor * curPix;
							sum += factor;
						}
					}
				}
				out[pos] = t / sum;
			}
		}
		TOCK("bilateralFilterKernel", size.x * size.y);
}

void depth2vertexKernel(float3* vertex, const float * depth, uint2 imageSize,
		const Matrix4 invK) {
	TICK();
	unsigned int x, y;
#pragma omp parallel for \
         shared(vertex), private(x, y)
	for (y = 0; y < imageSize.y; y++) {
		for (x = 0; x < imageSize.x; x++) {

			if (depth[x + y * imageSize.x] > 0) {
				vertex[x + y * imageSize.x] = depth[x + y * imageSize.x]
						* (rotate(invK, make_float3(x, y, 1.f)));
        }
			else {
				vertex[x + y * imageSize.x] = make_float3(0);
			}
		}
	}
	TOCK("depth2vertexKernel", imageSize.x * imageSize.y);
}

void rgb2intensity(float * grey_out, float3 *rgb_out, uint2 outSize, const uchar3 * in,
									 uint2 inSize) {
	TICK();
	// Check for unsupported conditions
	if ((inSize.x < outSize.x) || (inSize.y < outSize.y)) {
		std::cerr << "Invalid ratio." << std::endl;
		exit(1);
	}
	if ((inSize.x % outSize.x != 0) || (inSize.y % outSize.y != 0)) {
		std::cerr << "Invalid ratio." << std::endl;
		exit(1);
	}
	if ((inSize.x / outSize.x != inSize.y / outSize.y)) {
		std::cerr << "Invalid ratio." << std::endl;
		exit(1);
	}

	int ratio = inSize.x / outSize.x;
	unsigned int y;
#pragma omp parallel for shared(grey_out, rgb_out), private(y)
	for (y = 0; y < outSize.y; y++)
		for (unsigned int x = 0; x < outSize.x; x++) {
			const uchar3 rgb_resized = in[x * ratio + inSize.x * y * ratio];
			rgb_out[x + outSize.x * y].x = rgb_resized.x/255.0f;
			rgb_out[x + outSize.x * y].y = rgb_resized.y/255.0f;
			rgb_out[x + outSize.x * y].z = rgb_resized.z/255.0f;
			grey_out[x + outSize.x * y] = rgb2gs(rgb_resized);
		}
	TOCK("rgb2intensity", outSize.x * outSize.y);
}


void rgb2intensity(float * out, uint2 outSize, const uchar3 * in,
		uint2 inSize) {
	TICK();
	// Check for unsupported conditions
	if ((inSize.x < outSize.x) || (inSize.y < outSize.y)) {
		std::cerr << "Invalid ratio." << std::endl;
		exit(1);
	}
	if ((inSize.x % outSize.x != 0) || (inSize.y % outSize.y != 0)) {
		std::cerr << "Invalid ratio." << std::endl;
		exit(1);
	}
	if ((inSize.x / outSize.x != inSize.y / outSize.y)) {
		std::cerr << "Invalid ratio." << std::endl;
		exit(1);
	}

	int ratio = inSize.x / outSize.x;
	unsigned int y;
#pragma omp parallel for shared(out), private(y)
	for (y = 0; y < outSize.y; y++)
		for (unsigned int x = 0; x < outSize.x; x++) {
			out[x + outSize.x * y] = rgb2gs(in[x * ratio + inSize.x * y * ratio]);
		}
	TOCK("rgb2intensity", outSize.x * outSize.y);
}

void mm2metersKernel(float * out, uint2 outSize, const ushort * in,
		uint2 inSize) {
	TICK();
	// Check for unsupported conditions
	if ((inSize.x < outSize.x) || (inSize.y < outSize.y)) {
		std::cerr << "Invalid ratio." << std::endl;
		exit(1);
	}
	if ((inSize.x % outSize.x != 0) || (inSize.y % outSize.y != 0)) {
		std::cerr << "Invalid ratio." << std::endl;
		exit(1);
	}
	if ((inSize.x / outSize.x != inSize.y / outSize.y)) {
		std::cerr << "Invalid ratio." << std::endl;
		exit(1);
	}

	int ratio = inSize.x / outSize.x;
	unsigned int y;
#pragma omp parallel for \
        shared(out), private(y)
	for (y = 0; y < outSize.y; y++)
		for (unsigned int x = 0; x < outSize.x; x++) {
			out[x + outSize.x * y] = in[x * ratio + inSize.x * y * ratio]
					/ 1000.0f;
		}
	TOCK("mm2metersKernel", outSize.x * outSize.y);
}

void halfSampleRobustImageKernel(float* out, const float* in, uint2 inSize,
		const float e_d, const int r) {
	TICK();
	uint2 outSize = make_uint2(inSize.x / 2, inSize.y / 2);
	unsigned int y;
#pragma omp parallel for \
        shared(out), private(y)
	for (y = 0; y < outSize.y; y++) {
		for (unsigned int x = 0; x < outSize.x; x++) {
			uint2 pixel = make_uint2(x, y);
			const uint2 centerPixel = 2 * pixel;

			float sum = 0.0f;
			float t = 0.0f;
			const float center = in[centerPixel.x
					+ centerPixel.y * inSize.x];
			for (int i = -r + 1; i <= r; ++i) {
				for (int j = -r + 1; j <= r; ++j) {
					uint2 cur = make_uint2(
							clamp(
									make_int2(centerPixel.x + j,
											centerPixel.y + i), make_int2(0),
									make_int2(2 * outSize.x - 1,
											2 * outSize.y - 1)));
					float current = in[cur.x + cur.y * inSize.x];
					if (fabsf(current - center) < e_d) {
						sum += 1.0f;
						t += current;
					}
				}
			}
			out[pixel.x + pixel.y * outSize.x] = t / sum;
		}
	}
	TOCK("halfSampleRobustImageKernel", outSize.x * outSize.y);
}

/*
void GaussianDownsamplingKernel(float* out, const float* in, uint2 inSize) {
  TICK();
  float kernel[5][5] = {{1, 4, 6, 4, 1},
                         {4, 16, 24, 16, 4},
                         {6, 24, 36, 24, 6},
                         {4, 16, 24, 16, 4},
                         {1, 4, 6, 4, 1}};
  uint2 outSize = make_uint2(inSize.x / 2, inSize.y / 2);
  const int windowSize = 5;
  const int radius = (windowSize - 1)/2;
  unsigned int y;
#pragma omp parallel for \
        shared(out), private(y)
  for (y = 0; y < outSize.y; y++) {
    for (unsigned int x = 0; x < outSize.x; x++) {
      const int inX = 2*x+1;
      const int inY = 2*y+1;
      uint2 pixel = make_uint2(inX, inY);
      float sum = 0.0f;
      for (int a = 0; a < windowSize; a++) {
        for (int b = 0; b < windowSize; b++) {
          int xn = inX + a - radius;
          int yn = inY + b - radius;

          xn = min(inSize.x - 1, max(0, xn));
          yn = min(inSize.y - 1, max(0, yn));
          int index = xn + yn * inSize.x;
          sum += in[index] * kernel[a][b];
        }
        }
      out[x + y * outSize.x] = sum/16.0f;
    }
  }
  TOCK("halfSampleRobustImageKernel", outSize.x * outSize.y);
}
*/
void gradientsKernel(float * gradx, float * grady, const float * in, uint2 size) {
	TICK();
	unsigned int y;
#pragma omp parallel for private(y)
	for (y = 1; y < size.y-1; y++)
		for (unsigned int x = 1; x < size.x-1; x++) {
			gradx[x + size.x * y] = (in[x + 1 + size.x * y] - in[x - 1 + size.x * y]) * 0.5f;
			grady[x + size.x * y] = (in[x + size.x * (y + 1)] - in[x + size.x * (y - 1)]) * 0.5f;

		}
	TOCK("gradientsKernel", outSize.x * outSize.y);
}

/*
void SobelGradientsKernel(float * gradx, float * grady, const float * in, uint2 size, const int windowSize) {
	TICK();

	float kernelx[3][3] = {{-1, 0, 1},
						   {-2, 0, 2},
						   {-1, 0, 1}};

	float kernely[3][3] = {{-1, -2, -1},
						   {0,  0,  0},
						   {1, 2, 1}};
	unsigned int y;
    const int radius = (windowSize - 1)/2;
#pragma omp parallel private(y)
	for (y = 0; y < size.y; y++)
		for (unsigned int x = 0; x < size.x; x++)
        {
          //redundant?
          if (x < radius || x >= (size.x - radius) || y < radius || y >= (size.y - radius)) {
            gradx[x + size.x * y] = 0.f;
            grady[x + size.x * y] = 0.f;
          }
          else {
            float magX = 0.f;
            float magY = 0.f;
            for (int a = 0; a < windowSize; a++) {
              for (int b = 0; b < windowSize; b++) {
                int xn = x + a - radius;
                int yn = y + b - radius;

                int index = xn + yn * size.x;
                magX += in[index] * kernelx[a][b];
                magY += in[index] * kernely[a][b];
              }
            }
            gradx[x + size.x * y] = magX/8.0;
            grady[x + size.x * y] = magY/8.0;
          }
		}
	TOCK("gradientsKernel", outSize.x * outSize.y);
}
*/


void vertex2depth(float *rendered_depth, const float3 *vertex, const uint2 inputSize, const Matrix4& T_w_c) {
	TICK();
	unsigned int y;
#pragma omp parallel for shared(rendered_depth), private(y)
	for (y = 0; y < inputSize.y; y++)
#pragma simd
			for (unsigned int x = 0; x < inputSize.x; x++) {

				uint2 pos = make_uint2(x, y);
				uint idx = pos.x + pos.y * inputSize.x;

				if(vertex[idx].z > 0.0) {
					rendered_depth[idx] = (inverse(T_w_c) * vertex[idx]).z;
				} else {
					rendered_depth[idx] = 0.f;
				}
			}
}


Eigen::Quaternionf getQuaternion(const Matrix4& Trans){
	Eigen::Matrix3f rotationMatrix;
	rotationMatrix<<Trans.data[0].x,Trans.data[0].y, Trans.data[0].z,
			Trans.data[1].x,Trans.data[1].y,Trans.data[1].z,
			Trans.data[2].x,Trans.data[2].y,Trans.data[2].z;
	Eigen::Quaternionf q(rotationMatrix);
	return q;

}