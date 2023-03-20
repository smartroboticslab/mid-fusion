/*

 Copyright (c) 2011-2013 Gerhard Reitmayr, TU Graz

 Copyright (c) 2014 University of Edinburgh, Imperial College, University of Manchester.
 Developed in the PAMELA project, EPSRC Programme Grant EP/K008730/1

 This code is licensed under the MIT License.

 */

#ifndef _COMMONS_
#define _COMMONS_

#if defined(__GNUC__)
// circumvent packaging problems in gcc 4.7.0
#undef _GLIBCXX_ATOMIC_BUILTINS
#undef _GLIBCXX_USE_INT128

// need c headers for __int128 and uint16_t
#include <limits.h>
#endif
#include <sys/stat.h>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <string>
#include <cmath>
#include <iterator>
#include <set>

// Internal dependencies
#include <math_utils.h>
#include <voxel_traits.hpp>
#include <constant_parameters.h>

//External dependencies
#undef isnan
#undef isfinite
#include <TooN/TooN.h>
#include <TooN/se3.h>
#include <TooN/GR_SVD.h>
#include <tiny_obj_loader.h>
#include <Eigen/Dense>
////////////////////////// MATh STUFF //////////////////////
typedef Eigen::Matrix<float, 1, 80> All_Prob_Vect;
//typedef Eigen::VectorXf All_Prob_Vect;
//#define INVALID -2
// DATA TYPE
//

static const int INVALID = -2;

inline
bool is_file(std::string path) {
	struct stat buf;
	stat(path.c_str(), &buf);
	return S_ISREG(buf.st_mode);
}

template<typename T>
std::string NumberToString(T Number, int width = 6) {
	std::ostringstream ss;
	ss << std::setfill('0') << std::setw(width) << Number;
	return ss.str();
}

template<typename T>
void read_input(std::string inputfile, T * in) {
	size_t isize;
	std::ifstream file(inputfile.c_str(),
			std::ios::in | std::ios::binary | std::ios::ate);
	if (file.is_open()) {
		isize = file.tellg();
		file.seekg(0, std::ios::beg);
		file.read((char*) in, isize);
		file.close();
	} else {
		std::cout << "File opening failed : " << inputfile << std::endl;
		exit(1);
	}
}


inline uchar4 gs2rgb(double h) {
	uchar4 rgb;
	double v;
	double r, g, b;
	v = 0.75;
	if (v > 0) {
		double m;
		double sv;
		int sextant;
		double fract, vsf, mid1, mid2;
		m = 0.25;
		sv = 0.6667;
		h *= 6.0;
		sextant = (int) h;
		fract = h - sextant;
		vsf = v * sv * fract;
		mid1 = m + vsf;
		mid2 = v - vsf;
		switch (sextant) {
		case 0:
			r = v;
			g = mid1;
			b = m;
			break;
		case 1:
			r = mid2;
			g = v;
			b = m;
			break;
		case 2:
			r = m;
			g = v;
			b = mid1;
			break;
		case 3:
			r = m;
			g = mid2;
			b = v;
			break;
		case 4:
			r = mid1;
			g = m;
			b = v;
			break;
		case 5:
			r = v;
			g = m;
			b = mid2;
			break;
		default:
			r = 0;
			g = 0;
			b = 0;
			break;
		}
	}
	rgb.x = r * 255;
	rgb.y = g * 255;
	rgb.z = b * 255;
	rgb.w = 0; // Only for padding purposes 
	return rgb;
}

static inline float rgb2gs(const uchar3 rgb) {
  // as in https://docs.opencv.org/3.1.0/de/d25/imgproc_color_conversions.html
  // Normalised between [0,1]
  return (0.299f*rgb.x + 0.587f*rgb.y + 0.114f*rgb.z)/255.f;
}

typedef struct Triangle {
  float3 vertexes[3];
  float3 vnormals[3];
  float3 normal;
  float color;
  float surface_area;
  
  Triangle(){ 
    vertexes[0] = make_float3(0);
    vertexes[1] = make_float3(0);
    vertexes[2] = make_float3(0);
    normal = make_float3(0);
    surface_area = -1.f;
  }
  
  inline bool iszero(const float3& v){ 
    return (v.x == 0) && (v.y == 0) && (v.z == 0);
  }

  inline bool valid(){
    return !(iszero(vertexes[0]) && iszero(vertexes[1]) && iszero(vertexes[2]));
  }

  inline void compute_normal(){
    normal = cross(vertexes[1] - vertexes[0], vertexes[2] - vertexes[1]);
  }

  inline void compute_boundingbox(float3& minV, float3& maxV) const {
    minV = vertexes[0];
    maxV = vertexes[0];
    minV = fminf(minV, vertexes[0]);
    minV = fminf(minV, vertexes[1]);
    minV = fminf(minV, vertexes[2]);
    maxV = fmaxf(maxV, vertexes[0]);
    maxV = fmaxf(maxV, vertexes[1]);
    maxV = fmaxf(maxV, vertexes[2]);
  }

  inline float area() {
    // Use the cached value if available
    if(surface_area > 0) return surface_area;
    float3 a = vertexes[1] - vertexes[0];
    float3 b = vertexes[2] - vertexes[1];
    float3 v = cross(a,b);
    surface_area = (sqrtf(dot(v,v)))/2;
    return surface_area; 
  }

  float3 * uniform_sample(int num){

    float3 * points = new float3[num];
    for(int i = 0; i < num; ++i){
      float u = ((float)rand())/(float)RAND_MAX; 
      float v = ((float)rand())/(float)RAND_MAX;
      if(u + v > 1){
        u = 1 - u;
        v = 1 - v;
      }
      float w = 1 - (u + v);
      points[i] = u*vertexes[0] + v*vertexes[1] + w*vertexes[2];
    } 

    return points;
  }

  float3 * uniform_sample(int num, unsigned int& seed) const {

    float3 * points = new float3[num];
    for(int i = 0; i < num; ++i){
      float u = ((float)rand_r(&seed))/(float)RAND_MAX; 
      float v = ((float)rand_r(&seed))/(float)RAND_MAX;
      if(u + v > 1){
        u = 1 - u;
        v = 1 - v;
      }
      float w = 1 - (u + v);
      points[i] = u*vertexes[0] + v*vertexes[1] + w*vertexes[2];
    } 
    return points;
  }


// Triangle-box intersection first test: triangle's plane-bbox intersection
// Reference: Fast Parallel Surface and Solid Voxelization on GPUs
// http://research.michael-schwarz.com/publ/files/vox-siga10.pdf
inline bool planeOverlap(const float3& voxel, const float delta = 1.f) const {

  float3 c = make_float3(normal.x > 0 ? delta : 0,
                         normal.y > 0 ? delta : 0,
                         normal.z > 0 ? delta : 0);
  float d1 = dot(normal, c - vertexes[0]);
  float d2 = dot(normal, (delta - c) - vertexes[0]);
  
  return (dot(normal, voxel) + d1) * (dot(normal, voxel) + d2) <= 0;

}

// As in https://developer.nvidia.com/content/basics-gpu-voxelization
inline bool projectionOverlap(const float3& voxel, 
                               const float delta) const {

  float3 e0 = vertexes[1] - vertexes[0];
  float3 e1 = vertexes[2] - vertexes[1];
  float3 e2 = vertexes[0] - vertexes[2];
  float3 planeNormal = cross(e0, e1);

  // XY Plane
  
  {
    float isFront = std::signbit(planeNormal.z) ? 1 : -1;
    float2 eNrm[3];
    eNrm[0] = make_float2(e0.y, -e0.x) * isFront;
    eNrm[1] = make_float2(e1.y, -e1.x) * isFront;
    eNrm[2] = make_float2(e2.y, -e2.x) * isFront;

    float2 an[3];
    an[0] = fabs(eNrm[0]);
    an[1] = fabs(eNrm[1]);
    an[2] = fabs(eNrm[2]);

    float3 e0fs;
    e0fs.x = (an[0].x + an[0].y) * delta;
    e0fs.y = (an[1].x + an[1].y) * delta;
    e0fs.z = (an[2].x + an[2].y) * delta;

    float3 ef;
    float3 voxelCenter = voxel + delta/2;
    float2 voxelProj = make_float2(voxelCenter.x, voxelCenter.y);
    float2 v0 = make_float2(vertexes[0].x, vertexes[0].y);
    float2 v1 = make_float2(vertexes[1].x, vertexes[1].y);
    float2 v2 = make_float2(vertexes[2].x, vertexes[2].y);

    ef.x = e0fs.x - dot(v0 - voxelProj, eNrm[0]);
    ef.y = e0fs.y - dot(v1 - voxelProj, eNrm[1]);
    ef.z = e0fs.z - dot(v2 - voxelProj, eNrm[2]);

    if(ef.x < 0 || ef.y < 0 || ef.z < 0)
      return false;
  }

  // XZ Plane
  
  {
    float isFront = std::signbit(planeNormal.y) ? -1 : 1;
    float2 eNrm[3];
    eNrm[0] = make_float2(e0.z, -e0.x) * isFront;
    eNrm[1] = make_float2(e1.z, -e1.x) * isFront;
    eNrm[2] = make_float2(e2.z, -e2.x) * isFront;

    float2 an[3];
    an[0] = fabs(eNrm[0]);
    an[1] = fabs(eNrm[1]);
    an[2] = fabs(eNrm[2]);

    float3 e0fs;
    e0fs.x = (an[0].x + an[0].y) * delta;
    e0fs.y = (an[1].x + an[1].y) * delta;
    e0fs.z = (an[2].x + an[2].y) * delta;

    float3 ef;
    float3 voxelCenter = voxel + delta/2;
    float2 voxelProj = make_float2(voxelCenter.x, voxelCenter.z);
    float2 v0 = make_float2(vertexes[0].x, vertexes[0].z);
    float2 v1 = make_float2(vertexes[1].x, vertexes[1].z);
    float2 v2 = make_float2(vertexes[2].x, vertexes[2].z);

    ef.x = e0fs.x - dot(v0 - voxelProj, eNrm[0]);
    ef.y = e0fs.y - dot(v1 - voxelProj, eNrm[1]);
    ef.z = e0fs.z - dot(v2 - voxelProj, eNrm[2]);

    if(ef.x < 0 || ef.y < 0 || ef.z < 0)
      return false;
  }

  // PLANE YZ
  {
    float isFront = std::signbit(planeNormal.x) ? 1 : -1;
    float2 eNrm[3];
    eNrm[0] = make_float2(e0.z, -e0.y) * isFront;
    eNrm[1] = make_float2(e1.z, -e1.y) * isFront;
    eNrm[2] = make_float2(e2.z, -e2.y) * isFront;

    float2 an[3];
    an[0] = fabs(eNrm[0]);
    an[1] = fabs(eNrm[1]);
    an[2] = fabs(eNrm[2]);

    float3 e0fs;
    e0fs.x = (an[0].x + an[0].y) * delta;
    e0fs.y = (an[1].x + an[1].y) * delta;
    e0fs.z = (an[2].x + an[2].y) * delta;

    float3 ef;
    float3 voxelCenter = voxel + delta/2;
    float2 voxelProj = make_float2(voxelCenter.y, voxelCenter.z);
    float2 v0 = make_float2(vertexes[0].y, vertexes[0].z);
    float2 v1 = make_float2(vertexes[1].y, vertexes[1].z);
    float2 v2 = make_float2(vertexes[2].y, vertexes[2].z);

    ef.x = e0fs.x - dot(v0 - voxelProj, eNrm[0]);
    ef.y = e0fs.y - dot(v1 - voxelProj, eNrm[1]);
    ef.z = e0fs.z - dot(v2 - voxelProj, eNrm[2]);

    if(ef.x < 0 || ef.y < 0 || ef.z < 0)
      return false;
  }
  return true;  
}

// Implementation of http://www.geometrictools.com/Documentation/DistancePoint3Triangle3.pdf
// Reference source-code:
// http://www.mathworks.com/matlabcentral/fileexchange/22857-distance-between-a-point-and-a-triangle-in-3d
// and http://www.geometrictools.com/GTEngine/Include/Mathematics/GteDistPointTriangle.h
// Note: this method works but is rather ugly, there are alternatives.
// One might be the 2D method proposed in here: http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.477.6798&rep=rep1&type=pdf

inline float distance(const float3& point) const {

  const float3 B = vertexes[0];
  const float3 e0 = vertexes[1] - B;
  const float3 e1 = vertexes[2] - B;

  const float3 D = B - point;
  const float a = dot(e0, e0);
  const float b = dot(e0, e1);
  const float c = dot(e1, e1);
  const float d = dot(e0, D);
  const float e = dot(e1, D);
  // const float f = dot(D, D); Not sure why this is here and unused

  const float det = a*c - b*b;
  float s = b*e - c*d;
  float t = b*d - a*e;
 
  if(s+t <= det){
    if(s < 0){
      if(t < 0){
        // region 4
        if(d < 0){
          t = 0;
          if(-d >= a){
            s = 1;
          } 
          else{
            s = -d/a;
          }
        } 
        else{
          s = 0;
          if(e >= 0){
            t = 0;
          }
          else{
            if(-e >= c){
              t = 1;
            }
            else{
              t = -e/c;
            }
          }
        } 
      } // end of region 4
      else{
        // region 3
        s = 0;
        if(e >= 0){
          t = 0;
        }
        else{
          if(-e >= c){
            t = 1;
          }
          else{
            t = -e/c;
          }
        }
      } // end of region 3
    }
    else{
      if(t < 0){
        // region 5
        t = 0;
        if(d >= 0){
          s = 0;
        }
        else{
          if(-d >= a){
            s = 1;
          }        
          else{
            s = -d/a;
          }
        }
      }
      else{
        // region 0
        float invDet = 1/det;
        s = s*invDet;
        t = t*invDet;
      }
    }
  }
  else{ 
    if(s < 0){
      float tmp0 = b + d;
      float tmp1 = c + e;
      if(tmp1 > tmp0){
        float num = tmp1 - tmp0;
        float denom = a - 2*b +c;
        if(num >= denom){
          s = 1;
          t = 0;
         }
        else{
          s = num/denom;
          t = 1-s;
        }
      }
      else{
        s = 0;
        if(tmp1 <= 0){
          t = 1;
        }
        else{
          if(e >= 0){
            t = 0;
          }
          else{
            t = -e/c;
          }
        }
      }
    }
    else{
      if(t < 0){
        float tmp0 = b + e;
        float tmp1 = a + d;
        if(tmp1 > tmp0){
          float num = tmp1 - tmp0;
          float denom = a - 2*b + c;
          if(num >= denom){
            t = 1;
            s = 0;
          }
          else{
            t = num/denom;
            s = 1 - t;
          }
        }
        else{
          t = 0;
          if(tmp1 <= 0){
            s = 1;
          }
          else{
            if(d >= 0){
              s = 0;
            }
            else{
              s = -d/a;
            }
          }
        }
      }
      else{
        float num = c + e - b - d;
        if(num <= 0){
          s = 0;
          t = 1;
        }
        else{
          float denom = a - 2*b + c;
          if(num >= denom){
            s = 1;
            t = 0;
          }
          else{
            s = num/denom;
            t = 1-s;
          }
        }
      }
    }
  }
  const float3 closest = vertexes[0] + (s*e0) + (t*e1);
  float3 diff = point - closest;
  return sqrt(dot(diff, diff));
}

} Triangle;

struct RGBVolume {
    uint3 size;
    float3 dim;
    short4 * data;

    RGBVolume() { size = make_uint3(0); dim = make_float3(1); data = NULL; }

    __device__ float4 operator[]( const uint3 & pos ) const {
        const short4 d = data[pos.x + pos.y * size.x + pos.z * size.x * size.y];
        return make_float4(d.x * 0.00003051944088f, d.y * 0.00003051944088f, d.z * 0.00003051944088f, d.w); //  / 32766.0f
    }

    __device__ float3 v(const uint3 & pos) const {
        const float4 val = operator[](pos);
        return make_float3(val.x,val.y,val.z);
    }

    __device__ float3 vs(const uint3 & pos) const {
        const short4 val = data[pos.x + pos.y * size.x + pos.z * size.x * size.y];
        return make_float3(val.x,val.y,val.z);
    }

    __device__ void set(const uint3 & pos, const float4 & d ){
        data[pos.x + pos.y * size.x + pos.z * size.x * size.y] = make_short4(d.x * 32766.0f, d.y * 32766.0f, d.z * 32766.0f, d.w);
    }

    __device__ float3 pos( const uint3 & p ) const {
        return make_float3((p.x + 0.5f) * dim.x / size.x, (p.y + 0.5f) * dim.y / size.y, (p.z + 0.5f) * dim.z / size.z);
    }

    __device__ float3 interp( const float3 & pos ) const {
#if 0   // only for testing without linear interpolation
        const float3 scaled_pos = make_float3((pos.x * size.x / dim.x) , (pos.y * size.y / dim.y) , (pos.z * size.z / dim.z) );
        return v(make_uint3(clamp(make_int3(scaled_pos), make_int3(0), make_int3(size) - make_int3(1))));

#else
        const float3 scaled_pos = make_float3((pos.x * size.x / dim.x) - 0.5f, (pos.y * size.y / dim.y) - 0.5f, (pos.z * size.z / dim.z) - 0.5f);
        const int3 base = make_int3(floorf(scaled_pos));
        const float3 factor = fracf(scaled_pos);
        const int3 lower = max(base, make_int3(0));
        const int3 upper = min(base + make_int3(1), make_int3(size) - make_int3(1));
        return (
              ((vs(make_uint3(lower.x, lower.y, lower.z)) * (1-factor.x) + vs(make_uint3(upper.x, lower.y, lower.z)) * factor.x) * (1-factor.y)
             + (vs(make_uint3(lower.x, upper.y, lower.z)) * (1-factor.x) + vs(make_uint3(upper.x, upper.y, lower.z)) * factor.x) * factor.y) * (1-factor.z)
            + ((vs(make_uint3(lower.x, lower.y, upper.z)) * (1-factor.x) + vs(make_uint3(upper.x, lower.y, upper.z)) * factor.x) * (1-factor.y)
             + (vs(make_uint3(lower.x, upper.y, upper.z)) * (1-factor.x) + vs(make_uint3(upper.x, upper.y, upper.z)) * factor.x) * factor.y) * factor.z
            ) * 0.00003051944088f;
#endif
    }

    void init(uint3 s, float3 d){
        size = s;
        dim = d;
	 #ifdef CUDABASED
            cudaMalloc( &data,  size.x * size.y * size.z * sizeof(short4) ); 
         #else
	    std::cout << sizeof(short4) << std::endl;
	    data = (short4 *) malloc(size.x * size.y * size.z * sizeof(short4)); 
	    assert(data != (0));
         #endif
        //cudaMalloc(&data, size.x * size.y * size.z * sizeof(short4));
    }

    void release(){
      //cudaFree(data);
      //data = NULL;
           #ifdef ZEROCOPY_OPENCL
    	    clEnqueueUnmapMemObject(commandQueue, oclbuffer, data, 0, NULL, NULL);
    		clReleaseMemObject(oclbuffer);
		#else
        	free(data);
        	data = NULL;
		#endif // if def ZEROCOPY_OPENCL
    }
};


struct TrackData {
	int result;
	float error;
	float J[6];
};


inline Matrix4 getCameraMatrix(const float4 & k) {
	Matrix4 K;
	K.data[0] = make_float4(k.x, 0, k.z, 0);
	K.data[1] = make_float4(0, k.y, k.w, 0);
	K.data[2] = make_float4(0, 0, 1, 0);
	K.data[3] = make_float4(0, 0, 0, 1);
	return K;
}

inline Matrix4 getInverseCameraMatrix(const float4 & k) {
	Matrix4 invK;
	invK.data[0] = make_float4(1.0f / k.x, 0, -k.z / k.x, 0);
	invK.data[1] = make_float4(0, 1.0f / k.y, -k.w / k.y, 0);
	invK.data[2] = make_float4(0, 0, 1, 0);
	invK.data[3] = make_float4(0, 0, 0, 1);
	return invK;
}

inline Matrix4 inverse(const Matrix4 & A) {
	static TooN::Matrix<4, 4, float> I = TooN::Identity;
	TooN::Matrix<4, 4, float> temp = TooN::wrapMatrix<4, 4>(&A.data[0].x);
	Matrix4 R;
	TooN::wrapMatrix<4, 4>(&R.data[0].x) = TooN::gaussian_elimination(temp, I);
	return R;
}

inline Matrix4 operator*(const Matrix4 & A, const Matrix4 & B) {
	Matrix4 R;
	TooN::wrapMatrix<4, 4>(&R.data[0].x) = TooN::wrapMatrix<4, 4>(&A.data[0].x)
			* TooN::wrapMatrix<4, 4>(&B.data[0].x);
	return R;
}

static inline std::ostream& operator<<(std::ostream& os, const Matrix4& M) {
    os << "(" << M.data[0].x << ", " << M.data[0].y << ", " << M.data[0].z << ", " << M.data[0].w << ")" << std::endl;
    os << "(" << M.data[1].x << ", " << M.data[1].y << ", " << M.data[1].z << ", " << M.data[1].w << ")" << std::endl;
    os << "(" << M.data[2].x << ", " << M.data[2].y << ", " << M.data[2].z << ", " << M.data[2].w << ")" << std::endl;
    os << "(" << M.data[3].x << ", " << M.data[3].y << ", " << M.data[3].z << ", " << M.data[3].w << ")" << std::endl;
    return os;
}

template<typename P, typename A>
TooN::Matrix<6> makeJTJ(const TooN::Vector<21, P, A> & v) {
	TooN::Matrix<6> C = TooN::Zeros;
	C[0] = v.template slice<0, 6>();
	C[1].template slice<1, 5>() = v.template slice<6, 5>();
	C[2].template slice<2, 4>() = v.template slice<11, 4>();
	C[3].template slice<3, 3>() = v.template slice<15, 3>();
	C[4].template slice<4, 2>() = v.template slice<18, 2>();
	C[5][5] = v[20];

	for (int r = 1; r < 6; ++r)
		for (int c = 0; c < r; ++c)
			C[r][c] = C[c][r];

	return C;
}

template<typename T, typename A>
TooN::Vector<6> solve(const TooN::Vector<27, T, A> & vals) {
	const TooN::Vector<6> b = vals.template slice<0, 6>();
	const TooN::Matrix<6> C = makeJTJ(vals.template slice<6, 21>());

	TooN::GR_SVD<6, 6> svd(C);
	return svd.backsub(b, 1e6);
}

template<typename P>
inline Matrix4 toMatrix4(const TooN::SE3<P> & p) {
	const TooN::Matrix<4, 4, float> I = TooN::Identity;
	Matrix4 R;
	TooN::wrapMatrix<4, 4>(&R.data[0].x) = p * I;
	return R;
}

inline Matrix4 Identity() {
  return toMatrix4(TooN::SE3<>(TooN::Vector<6>(TooN::Zeros)));
}

static const float epsilon = 0.0000001;

inline void compareTrackData(std::string str, TrackData* l, TrackData * r,
		uint size) {
	for (unsigned int i = 0; i < size; i++) {
		if (std::abs(l[i].error - r[i].error) > epsilon) {
			std::cout << "Error into " << str << " at " << i << std::endl;
			std::cout << "l.error =  " << l[i].error << std::endl;
			std::cout << "r.error =  " << r[i].error << std::endl;
		}

		if (std::abs(l[i].result - r[i].result) > epsilon) {
			std::cout << "Error into " << str << " at " << i << std::endl;
			std::cout << "l.result =  " << l[i].result << std::endl;
			std::cout << "r.result =  " << r[i].result << std::endl;
		}

	}
}

inline void compareFloat(std::string str, float* l, float * r, uint size) {
	for (unsigned int i = 0; i < size; i++) {
		if (std::abs(l[i] - r[i]) > epsilon) {
			std::cout << "Error into " << str << " at " << i << std::endl;
			std::cout << "l =  " << l[i] << std::endl;
			std::cout << "r =  " << r[i] << std::endl;
		}
	}
}
inline void compareFloat3(std::string str, float3* l, float3 * r, uint size) {
	for (unsigned int i = 0; i < size; i++) {
		if (std::abs(l[i].x - r[i].x) > epsilon) {
			std::cout << "Error into " << str << " at " << i << std::endl;
			std::cout << "l.x =  " << l[i].x << std::endl;
			std::cout << "r.x =  " << r[i].x << std::endl;
		}
		if (std::abs(l[i].y - r[i].y) > epsilon) {
			std::cout << "Error into " << str << " at " << i << std::endl;
			std::cout << "l.y =  " << l[i].y << std::endl;
			std::cout << "r.y =  " << r[i].y << std::endl;
		}
		if (std::abs(l[i].z - r[i].z) > epsilon) {
			std::cout << "Error into " << str << " at " << i << std::endl;
			std::cout << "l.z =  " << l[i].z << std::endl;
			std::cout << "r.z =  " << r[i].z << std::endl;
		}
	}
}

inline void compareFloat4(std::string str, float4* l, float4 * r, uint size) {
	for (unsigned int i = 0; i < size; i++) {
		if (std::abs(l[i].x - r[i].x) > epsilon) {
			std::cout << "Error into " << str << " at " << i << std::endl;
			std::cout << "l.x =  " << l[i].x << std::endl;
			std::cout << "r.x =  " << r[i].x << std::endl;
		}
		if (std::abs(l[i].y - r[i].y) > epsilon) {
			std::cout << "Error into " << str << " at " << i << std::endl;
			std::cout << "l.y =  " << l[i].y << std::endl;
			std::cout << "r.y =  " << r[i].y << std::endl;
		}
		if (std::abs(l[i].z - r[i].z) > epsilon) {
			std::cout << "Error into " << str << " at " << i << std::endl;
			std::cout << "l.z =  " << l[i].z << std::endl;
			std::cout << "r.z =  " << r[i].z << std::endl;
		}
		if (std::abs(l[i].w - r[i].w) > epsilon) {
			std::cout << "Error into " << str << " at " << i << std::endl;
			std::cout << "l.w =  " << l[i].w << std::endl;
			std::cout << "r.w =  " << r[i].w << std::endl;
		}
	}
}

inline void compareMatrix4(std::string str, Matrix4 l, Matrix4 r) {
	compareFloat4(str, l.data, r.data, 4);
}

inline bool compareFloat4(float4* l, float4 * r, uint size) {
	for (unsigned int i = 0; i < size; i++) {
		if ((std::abs(l[i].x - r[i].x) > epsilon) ||
		    (std::abs(l[i].y - r[i].y) > epsilon) ||
		    (std::abs(l[i].z - r[i].z) > epsilon) ||
		    (std::abs(l[i].w - r[i].w) > epsilon))
     return false; 
	}
  return true;
}

inline bool compareMatrix4(Matrix4 l, Matrix4 r) {
	return compareFloat4(l.data, r.data, 4);
}

inline void printMatrix4(std::string str, Matrix4 l) {
	std::cout << "printMatrix4 : " << str << std::endl;
	for (int i = 0; i < 4; i++) {
		std::cout << "  [" << l.data[i].x << "," << l.data[i].y << ","
				<< l.data[i].z << "," << l.data[i].w << "]" << std::endl;
	}
}
inline void compareNormal(std::string str, float3* l, float3 * r, uint size) {
	for (unsigned int i = 0; i < size; i++) {
		if (std::abs(l[i].x - r[i].x) > epsilon) {
			std::cout << "Error into " << str << " at " << i << std::endl;
			std::cout << "l.x =  " << l[i].x << std::endl;
			std::cout << "r.x =  " << r[i].x << std::endl;
		} else if (r[i].x != INVALID) {
			if (std::abs(l[i].y - r[i].y) > epsilon) {
				std::cout << "Error into " << str << " at " << i << std::endl;
				std::cout << "l.y =  " << l[i].y << std::endl;
				std::cout << "r.y =  " << r[i].y << std::endl;
			}
			if (std::abs(l[i].z - r[i].z) > epsilon) {
				std::cout << "Error into " << str << " at " << i << std::endl;
				std::cout << "l.z =  " << l[i].z << std::endl;
				std::cout << "r.z =  " << r[i].z << std::endl;
			}
		}
	}
}

template<typename T>
void writefile(std::string prefix, int idx, T * data, uint size) {

	std::string filename = prefix + NumberToString(idx);
	FILE* pFile = fopen(filename.c_str(), "wb");

	if (!pFile) {
		std::cout << "File opening failed : " << filename << std::endl;
		exit(1);
	}

	size_t write_cnt = fwrite(data, sizeof(T), size, pFile);

	std::cout << "File " << filename << " of size " << write_cnt << std::endl;

	fclose(pFile);
}

template<typename T>
void writefile(std::string prefix, int idx, T * data, uint2 size) {
	writefile(prefix, idx, data, size.x * size.y);
}
inline
void writeposfile(std::string prefix, int idx, Matrix4 m, uint) {

	writefile("BINARY_" + prefix, idx, m.data, 4);

	std::string filename = prefix + NumberToString(idx);
	std::ofstream pFile;
	pFile.open(filename.c_str());

	if (pFile.fail()) {
		std::cout << "File opening failed : " << filename << std::endl;
		exit(1);
	}

	pFile << m.data[0].x << " " << m.data[0].y << " " << m.data[0].z << " "
			<< m.data[0].w << std::endl;
	pFile << m.data[1].x << " " << m.data[1].y << " " << m.data[1].z << " "
			<< m.data[1].w << std::endl;
	pFile << m.data[2].x << " " << m.data[2].y << " " << m.data[2].z << " "
			<< m.data[2].w << std::endl;
	pFile << m.data[3].x << " " << m.data[3].y << " " << m.data[3].z << " "
			<< m.data[3].w << std::endl;

	std::cout << "Pose File " << filename << std::endl;

	pFile.close();
}

// void writeVolume(std::string filename, Volume v) {
// 
// 	std::ofstream fDumpFile;
// 	fDumpFile.open(filename.c_str(), std::ios::out | std::ios::binary);
// 
// 	if (fDumpFile.fail()) {
// 		std::cout << "Error opening file: " << filename << std::endl;
// 		exit(1);
// 	}
// 
// 	// Retrieve the volumetric representation data
// 	short2 *hostData = (short2 *) v.data;
// 
// 	// Dump on file without the y component of the short2 variable
// 	for (unsigned int i = 0; i < v.size.x * v.size.y * v.size.z; i++) {
// 		fDumpFile.write((char *) (hostData + i), sizeof(short));
// 	}
// 
// 	fDumpFile.close();
// }

// Load .obj file int tinyobj representation.
inline bool loadObjFile(std::string filename, 
                        std::vector<tinyobj::shape_t>& shapes, 
                        std::vector<tinyobj::material_t>& materials){

  std::string err;
  bool ret = tinyobj::LoadObj(shapes, materials, err, filename.c_str(), "/");
  if(!ret){
    std::cout << "Error while loading .obj file: " << err <<  std::endl;
  }
  return ret;
}

inline void objToTriangles(const std::vector<tinyobj::shape_t>& shapes,
                           std::vector<Triangle>& out){
  for(unsigned int i = 0; i < shapes.size(); ++i){
    const tinyobj::shape_t & shape = shapes[i];  
    
    const std::vector<unsigned int>& indices = shape.mesh.indices;    
    for(unsigned int it = 0; it < shape.mesh.indices.size() / 3; ++it){
      Triangle t;
      uint index = (indices[it*3]);
      
      t.vertexes[0] = make_float3(shape.mesh.positions.at(3*index+0),
                                  shape.mesh.positions.at(3*index+1),
                                  shape.mesh.positions.at(3*index+2));

      index = indices[it*3+1];
      t.vertexes[1] = make_float3(shape.mesh.positions.at(3*index+0),
                                  shape.mesh.positions.at(3*index+1),
                                  shape.mesh.positions.at(3*index+2));
      index = indices[it*3+2];
      t.vertexes[2] = make_float3(shape.mesh.positions.at(3*index+0),
                                  shape.mesh.positions.at(3*index+1),
                                  shape.mesh.positions.at(3*index+2));
     
      t.compute_normal(); 
      t.area();
      out.push_back(t);
    } 
  }
}

inline void writeVtkMesh(const char * filename, 
                         const std::vector<Triangle>& mesh,
                         const float * point_data = NULL,
                         const float * cell_data = NULL){
  std::stringstream points;
  std::stringstream polygons;
  std::stringstream pointdata;
  std::stringstream celldata;
  int point_count = 0;
  int triangle_count = 0;
  bool hasPointData = point_data != NULL;
  bool hasCellData = cell_data != NULL;

  for(unsigned int i = 0; i < mesh.size(); ++i ){
    const Triangle& t = mesh[i];

    points << t.vertexes[0].x << " " << t.vertexes[0].y << " " 
      << t.vertexes[0].z << std::endl; 
    points << t.vertexes[1].x << " " << t.vertexes[1].y << " " 
      << t.vertexes[1].z << std::endl; 
    points << t.vertexes[2].x << " " << t.vertexes[2].y << " " 
      << t.vertexes[2].z << std::endl; 

    polygons << "3 " << point_count << " " << point_count+1 << 
      " " << point_count+2 << std::endl;

    if(hasPointData){
      pointdata << point_data[i*3] << std::endl;
      pointdata << point_data[i*3 + 1] << std::endl;
      pointdata << point_data[i*3 + 2] << std::endl;
    }

    if(hasCellData){
      celldata << cell_data[i] << std::endl;
    }

    point_count +=3;
    triangle_count++;
  }   

  std::ofstream f;
  f.open(filename);
  f << "# vtk DataFile Version 1.0" << std::endl;
  f << "vtk mesh generated from KFusion" << std::endl;
  f << "ASCII" << std::endl;
  f << "DATASET POLYDATA" << std::endl;

  f << "POINTS " << point_count << " FLOAT" << std::endl;
  f << points.str();

  f << "POLYGONS " << triangle_count << " " << triangle_count * 4 << std::endl;
  f << polygons.str() << std::endl;
  if(hasPointData){
    f << "POINT_DATA " << point_count << std::endl; 
    f << "SCALARS vertex_scalars float 1" << std::endl;
    f << "LOOKUP_TABLE default" << std::endl;
    f << pointdata.str();
  }

  if(hasCellData){
    f << "CELL_DATA " << triangle_count << std::endl; 
    f << "SCALARS cell_scalars float 1" << std::endl;
    f << "LOOKUP_TABLE default" << std::endl;
    f << celldata.str();
  }
  f.close();
}

inline void writeObjMesh(const char * filename,
                         const std::vector<Triangle>& mesh){
  std::stringstream points;
  std::stringstream faces;
  int point_count = 0;
  int face_count = 0;

  for(unsigned int i = 0; i < mesh.size(); i++){
    const Triangle& t = mesh[i];  
    points << "v " << t.vertexes[0].x << " " << t.vertexes[0].y
           << " "  << t.vertexes[0].z << std::endl;
    points << "v " << t.vertexes[1].x << " " << t.vertexes[1].y 
           << " "  << t.vertexes[1].z << std::endl;
    points << "v " << t.vertexes[2].x << " " << t.vertexes[2].y 
           << " "  << t.vertexes[2].z << std::endl;

    faces  << "f " << (face_count*3)+1 << " " << (face_count*3)+2 
           << " " << (face_count*3)+3 << std::endl;

    point_count +=3;
    face_count += 1;
  }

  std::ofstream f(filename); 
  f << "# OBJ file format with ext .obj" << std::endl;
  f << "# vertex count = " << point_count << std::endl;
  f << "# face count = " << face_count << std::endl;
  f << points.str();
  f << faces.str();
  f.close();
  std::cout << "Written " << face_count << " faces and " << point_count 
            << " points" << std::endl;
}

#endif
