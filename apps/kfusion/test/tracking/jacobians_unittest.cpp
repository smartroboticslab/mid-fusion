#include "math_utils.h"
#include "commons.h"
#include "gtest/gtest.h"
#include "../../src/cpp/tracking/tracking.h"
#include "../../src/cpp/preprocessing/preprocessing.h"
//#include "../src/cpp/rendering.cpp"
#include "lodepng.h"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <sstream>
#include <math.h>       /* fabs */

/* Annoying stat declaration to please benchmarking code */
#include "../../../utils/perfstats.h"
PerfStats Stats;

class JacobianTest: public ::testing::Test {
 protected:
  TrackingPointer rgbTracker;
  virtual void SetUp() {
    unsigned error;
    uchar4* l_rgb;
    uchar4* r_rgb;
    uchar3* temp_buff;
    unsigned char * raw_r_depth;
    unsigned char * raw_l_depth;

    /* FILEPATH: absolute path of this test source file, defined in CMakeLists.txt */
/////////////    icl
//    error = lodepng_decode32_file((unsigned char **)&l_rgb, &framesize.x, &framesize.y, FILEPATH"/721.png");
//    ASSERT_EQ(error, 0);
//    error = lodepng_decode32_file((unsigned char **)&r_rgb, &framesize.x, &framesize.y, FILEPATH"/720.png");
//    ASSERT_EQ(error, 0);
//    error = lodepng_decode_file((unsigned char **)&raw_l_depth, &framesize.x,
//        &framesize.y, FILEPATH"/721_depth.png",
//        LodePNGColorType::LCT_GREY, 16); // same defaults as in cpp bindings
//    ASSERT_EQ(error, 0);
//    error = lodepng_decode_file((unsigned char **)&raw_r_depth, &framesize.x, &framesize.y, FILEPATH"/720_depth.png",
//        LodePNGColorType::LCT_GREY, 16); // same defaults as in cpp bindings
//    ASSERT_EQ(error, 0);
//      //// ICL
//    GT_r = toMatrix4({0.0494376, -0.487105, 0.081092, 0.868164}, {-0.0236373, -0.169094, -1.79256});
//    GT_l = toMatrix4({0.0512001, -0.484772, 0.080838, 0.86939}, {-0.0319467, -0.171483, -1.78783});
//    k = {481.20, -480, 319.50, 239.50};//icl

////    tum-fr3-walking-halfsphere
    error = lodepng_decode32_file((unsigned char **) &r_rgb, &framesize.x, &framesize.y, FILEPATH
    "/1341846436.889894.png");
    ASSERT_EQ(error, 0);
    error = lodepng_decode32_file((unsigned char **) &l_rgb, &framesize.x, &framesize.y, FILEPATH
    "/1341846436.922139.png");
    ASSERT_EQ(error, 0);
    error = lodepng_decode_file((unsigned char **)&raw_r_depth, &framesize.x,
                                &framesize.y, FILEPATH"/1341846436.889927.png",
        LodePNGColorType::LCT_GREY, 16); // same defaults as in cpp bindings
    ASSERT_EQ(error, 0);
    error = lodepng_decode_file((unsigned char **)&raw_l_depth, &framesize.x,
                                &framesize.y, FILEPATH"/1341846436.922153.png",
        LodePNGColorType::LCT_GREY, 16); // same defaults as in cpp bindings
    ASSERT_EQ(error, 0);

    // // //  TUM
    GT_r = toMatrix4({0.0068, -2.7651, 1.6077, -0.7091}, {-0.2017, 0.2146, 0.6407});
    GT_l = toMatrix4({0.0167, -2.7562, 1.6066, -0.7062}, {-0.2011, 0.2219, 0.6416});
    k = {525.0, 525.0, 319.5, 239.5};//tum fr-ros

////
    l_image = (float *) malloc(sizeof(float) * framesize.x * framesize.y);
    r_image = (float *) malloc(sizeof(float) * framesize.x * framesize.y);
    l_gradx = (float *) malloc(sizeof(float) * framesize.x * framesize.y);
    l_grady = (float *) malloc(sizeof(float) * framesize.x * framesize.y);
    r_depthf = (float *) malloc(sizeof(float) * framesize.x * framesize.y);
    l_depthf = (float *) malloc(sizeof(float) * framesize.x * framesize.y);
    temp_buff = (uchar3 *) malloc(sizeof(uchar3) * framesize.x * framesize.y);
    outbuffer = (uchar4 *) calloc(sizeof(uchar4) * framesize.x * framesize.y, 1);
    scaled_r_depth = (float **) calloc(sizeof(float*) * pyramid_levels, 1);
    scaled_r_vertex = (float3 **) calloc(sizeof(float3*) * pyramid_levels, 1);
    scaled_l_image = (float **)  calloc(sizeof(float*) * pyramid_levels, 1);
    scaled_r_image = (float **) calloc(sizeof(float*) * pyramid_levels, 1);
    scaled_l_gradx = (float **) calloc(sizeof(float *) * framesize.x * framesize.y, 1);
    scaled_l_grady = (float **) calloc(sizeof(float *) * framesize.x * framesize.y, 1);


    for (unsigned int i = 0; i < framesize.y * framesize.x; i++)
      temp_buff[i] = make_uchar3(l_rgb[i].x, l_rgb[i].y, l_rgb[i].z);
    rgb2intensity(l_image, framesize, temp_buff, framesize);

    for (unsigned int i = 0; i < framesize.y * framesize.x; i++)
      temp_buff[i] = make_uchar3(r_rgb[i].x, r_rgb[i].y, r_rgb[i].z);
    rgb2intensity(r_image, framesize, temp_buff, framesize);

    int i = 0;
    for(unsigned y = 0; y < framesize.y; y++)
      for (unsigned x = 0; x < framesize.x; x++) {
        size_t index = y * framesize.x * 2 + x * 2;
        int r = raw_r_depth[index + 0] * 256 + raw_r_depth[index + 1];
        int l = raw_l_depth[index + 0] * 256 + raw_l_depth[index + 1];
        r_depthf[i++] = r/5000.f; // convert to meters
        l_depthf[i++] = l/5000.f; // convert to meters
      }

    // renderDepthKernel(outbuffer, r_depthf, framesize, 0.4f, 4.0f);
    // std::stringstream filename;
    // filename << FILEPATH"/depth.png";
    // if(unsigned int error = lodepng_encode32_file(filename.str().c_str(),
    //       (unsigned char *)outbuffer, framesize.x, framesize.y))
    //   printf("error %u: %s\n", error, lodepng_error_text(error));
    // filename.str(std::string());

    /* Precompute image gradient */
    gradientsKernel(l_gradx, l_grady, l_image, framesize);
//    SobelGradientsKernel(gradx, grady, right, framesize, 3);

    /* Get vertex array */
    r_vertex = (float3 * ) malloc(sizeof(float3) * framesize.x * framesize.y);
    K = getCameraMatrix(k);
    depth2vertexKernel(r_vertex, r_depthf, framesize, inverse(K));

    std::vector<int> iterations;
    iterations.clear();
    for (int level = 0; level != pyramid_levels; level++) {
      iterations.push_back(pyramid_iter[level]);
    }
    rgbTracker = std::make_shared<Tracking>(framesize, iterations, true, false, true);

  }


  float * l_image; // tracked frame
  float * r_image; // reference frame
  float * l_gradx;
  float * l_grady;
  float * r_depthf; // raw depth of reference frame
  float * l_depthf;
  float ** scaled_l_gradx;
  float ** scaled_l_grady;
  float ** scaled_r_depth;
  float ** scaled_l_image;
  float ** scaled_r_image;
  float3 * r_vertex; // vertex map obtained from depthf
  float3 ** scaled_r_vertex; // vertex map obtained from depthf
  uchar4 * outbuffer;
  TooN::Vector<640*480, float> numerical_J;
  TooN::Vector<640*480, float> analytic_J;
  uint2 framesize = {640, 480};

  float4 k;
  Matrix4 K;
  int pyramid_levels = 3;
  int pyramid_iter[3] = {20, 20, 20};
  float min_grad = - INFINITY;
  Matrix4 GT_r, GT_l;

  const float rgb_tracking_threshold_[3] = {100, 100, 100};
//  const float mini_gradient_magintude[3] = {0.5/255., 0.3/255., 0.1/255.};
  const float mini_gradient_magintude[3] = {0./255., 0./255., 0./255.};
};

TEST_F(JacobianTest, IdentitySameImage) {

  const float identity[6] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
  const Matrix4 I = toMatrix4(TooN::SE3<>(TooN::Vector<6, const float, TooN::Reference>(identity)));
  const Matrix4 invK = inverse(K);
  for(unsigned y = 0; y < framesize.y; y++)
    for (unsigned x = 0; x < framesize.x; x++) {
      float depth = r_depthf[x + framesize.x*y];
      if(depth == 0.f)
        continue;
      float3 r_test = depth * rotate(invK, make_float3(x, y, 1.f));
      float3 w_test = r_test;
      float3 l_test = r_test;
      const float3 invTest = rotate(K, l_test);
      const float2 testpix = make_float2(invTest.x/invTest.z, invTest.y/invTest.z);
      const float3 r_v = r_vertex[x + framesize.x*y];
      const float2 projpixel = rgbTracker->warp(l_test, w_test, I, I, K, r_v);
      ASSERT_FLOAT_EQ(testpix.x, projpixel.x);
      ASSERT_FLOAT_EQ(testpix.y, projpixel.y);
      if (projpixel.x < 1  || projpixel.x > framesize.x - 2
          || projpixel.y < 1 || projpixel.y > framesize.y - 2) {
        continue;
      }
      const uint2 r_pixel = make_uint2(x, y);
      const float r = rgbTracker->rgb_residual(r_image, r_pixel, framesize,
                                               l_image, projpixel, framesize);
      ASSERT_NEAR(r, 0.f, 1e-4f);
    }
}

void renderError(cv::Mat& errorImg, const TrackData* data, const uint2 outSize, const uint2 jacobianSize){
  errorImg = cv::Mat::zeros(cv::Size(outSize.x, outSize.y), CV_8UC1);

  unsigned int y;
#pragma omp parallel for \
        shared(errorImg), private(y)
  for (y = 0; y < outSize.y; y++)
    for (unsigned int x = 0; x < outSize.x; x++) {
      uint pos = x + y * jacobianSize.x;
      if (data[pos].result <1){
        continue;
      }
      float e = data[pos].error;
      errorImg.at<uchar>(y,x) = fabs(255.0 * e);
    }
}

void renderImage(cv::Mat& errorImg, const float* inImage, const uint2 outSize){
  errorImg = cv::Mat::zeros(cv::Size(outSize.x, outSize.y), CV_8UC1);

  unsigned int y;
#pragma omp parallel for \
        shared(errorImg), private(y)
  for (y = 0; y < outSize.y; y++)
    for (unsigned int x = 0; x < outSize.x; x++) {
      uint pos = x + y * outSize.x;
      float e = inImage[pos];
      e = 255.0 * e;

      errorImg.at<uchar>(y,x) = fabs(e);
    }
}

TEST_F(JacobianTest, JacobianTest) {

  float JT[framesize.x*framesize.y][6];
  float perturbation = 1e-5f;
  for(int i = 0; i < 6 ; ++i) {
    int evaluated_pixels = 0;
    for(unsigned y = 0; y < framesize.y; y++) {
      for (unsigned x = 0; x < framesize.x; x++) {
        const uint2 r_pix = make_uint2(x, y);
        const unsigned idx = x + y*framesize.x;
        if(r_depthf[idx] == 0.f)
          continue;

        float perturbed[6] = {0.f, 0.f, 0.f, 0.f,  0.f, 0.f};
        perturbed[i] = perturbation;
        const float identity[6] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
        const Matrix4 I = toMatrix4(TooN::SE3<>(TooN::Vector<6, const float, TooN::Reference>(identity)));
        const Matrix4 Tplus = toMatrix4(TooN::SE3<>(TooN::Vector<6, const float, TooN::Reference>(perturbed)));
        const Matrix4 Tminus = toMatrix4(TooN::SE3<>(-1.f * TooN::Vector<6, const float, TooN::Reference>(perturbed)));

        const float3 r_v = I * r_vertex[r_pix.x + framesize.x*r_pix.y];
        float3 l_v_warped, l_v_warped_p, l_v_warped_m;
        float3 w_v_warped, w_v_warped_p, w_v_warped_m;
        const float2 projpixel = rgbTracker->warp(l_v_warped, w_v_warped, I, I, K, r_v);
        const float2 projpixel_p = rgbTracker->warp(l_v_warped_p, w_v_warped_p, I, Tplus,
                                                    K, r_v);
        const float2 projpixel_m = rgbTracker->warp(l_v_warped_m, w_v_warped_m, I,
                                                    Tminus, K, r_v);
        if (projpixel.x < 1  || projpixel.x > framesize.x - 2 || projpixel.y < 1
            || projpixel.y > framesize.y - 2 || projpixel_p.x < 1
            || projpixel_p.x > framesize.x - 2 || projpixel_p.y < 1
            || projpixel_p.y > framesize.y - 2 || projpixel_m.x < 1
            || projpixel_m.x > framesize.x - 2 || projpixel_m.y < 1
            || projpixel_m.y > framesize.y - 2) {
          continue;
        }

        const uint2 warped = make_uint2(projpixel.x, projpixel.y);

        const float r_t = rgbTracker->rgb_residual(r_image, r_pix, framesize, l_image,
                                                   projpixel_p, framesize);
        const float r_m = rgbTracker->rgb_residual(r_image, r_pix, framesize, l_image,
                                                   projpixel_m, framesize);

        float * J = JT[idx];
        bool calculated = rgbTracker->rgb_jacobian(J, l_v_warped, w_v_warped, I, projpixel,
                                  l_gradx, l_grady, framesize, K, min_grad, 1.0f);
//        if (!calculated) continue;
        float fin_diff = (r_t - r_m)/(2.f*perturbation);
        /* Negating the J as this is not done in rgb_jacobians */
//        diff[idx] = fin_diff - (- 1.f * J[i]);
        analytic_J[idx] = (- 1.f * J[i]);
        numerical_J[idx] = fin_diff;
        evaluated_pixels++;
      }
    }

//    float norm = TooN::norm_1(diff);
    float analtic_Jacobian = TooN::norm_1(analytic_J)/(evaluated_pixels);
    float numerical_Jacobian = TooN::norm_1(numerical_J)/(evaluated_pixels);
    float RMSE_percent = fabsf(analtic_Jacobian-numerical_Jacobian)/numerical_Jacobian*100.0;
    std::cout<<i+1<<"-th component numerical jacobian: "<<numerical_Jacobian<<std::endl;
    std::cout<<i+1<<"-th component analytic jacobian: "<<analtic_Jacobian<<std::endl;
    std::cout<<"error: "<<RMSE_percent<<"%"<<std::endl;
//    float RMSE = norm/(framesize.x*framesize.y);
    EXPECT_LE(RMSE_percent, 1.0f);
    for(unsigned k = 0; k < framesize.x*framesize.y; ++k){
      analytic_J[k] = 0.f;
      numerical_J[k] = 0.f;
    }
  }
}


void ICP_Kernel_Jacobian_test(TrackData *output,
                              const uint2 jacobian_size,
                              const float3 *inVertex,
                              const float3 *inNormal,
                              uint2 inSize,
                              const float3 *refVertex,
                              const float3 *refNormal,
                              uint2 refSize,
                              const Matrix4& Ttrack,
                              const Matrix4& view,
                              const float dist_threshold,
                              const float normal_threshold,
                              const float3 *icp_cov_layer) {
  TICK();
  uint2 pixel = make_uint2(0, 0);

  float perturbation = 1e-5f;

  for (int i = 3; i < 6; ++i) {
    float perturbed[6] = {0.f, 0.f, 0.f, 0.f,  0.f, 0.f};
    perturbed[i] = perturbation;
    float evaluated_pixels = 0;
    unsigned int pixely, pixelx;
    float sum_diff = 0.f;
    for (pixely = 0; pixely < inSize.y; pixely++) {
      for (pixelx = 0; pixelx < inSize.x; pixelx++) {
        pixel.x = pixelx;
        pixel.y = pixely;

        const unsigned idx = pixel.x + pixel.y * inSize.x;
        TrackData &row = output[pixel.x + pixel.y * jacobian_size.x];

        if (inNormal[pixel.x + pixel.y * inSize.x].x == INVALID) {
          row.result = -1;
          continue;
        }


        const Matrix4 p_plus = toMatrix4(TooN::SE3<>(TooN::Vector<6, const
        float, TooN::Reference>(perturbed)));
        const Matrix4 p_minus = toMatrix4(TooN::SE3<>(-1.f * TooN::Vector<6,
            const float, TooN::Reference>(perturbed)));
        const Matrix4 Tplus = p_plus * Ttrack;
        const Matrix4 Tminus = p_minus * Ttrack;

        const float3 projectedVertex = Ttrack
            * inVertex[pixel.x + pixel.y * inSize.x];
        const float3 projectedPos = view * projectedVertex;

        const float3 projectedVertex_p = Tplus
            * inVertex[pixel.x + pixel.y * inSize.x];

        const float3 projectedVertex_m = Tminus
            * inVertex[pixel.x + pixel.y * inSize.x];

        const float2 projPixel = make_float2(
            projectedPos.x / projectedPos.z,
            projectedPos.y / projectedPos.z);

        if (projPixel.x < 1 || projPixel.x > refSize.x - 2
            || projPixel.y < 1 || projPixel.y > refSize.y - 2) {
          row.result = -2;
          continue;
        }

        const uint2 refPixel = make_uint2(projPixel.x, projPixel.y);
        const float3 referenceNormal = refNormal[refPixel.x
            + refPixel.y * refSize.x];

        if (referenceNormal.x == INVALID ) {
          row.result = -3;
          continue;
        }


        const float3 diff = refVertex[refPixel.x + refPixel.y * refSize.x]
            - projectedVertex;
        const float3 diff_p = refVertex[refPixel.x + refPixel.y * refSize.x]
            - projectedVertex_p;
        const float3 diff_m = refVertex[refPixel.x + refPixel.y * refSize.x]
            - projectedVertex_m;
        const float3 projectedNormal =
            rotate(Ttrack, inNormal[pixel.x + pixel.y * inSize.x]);

        if ((length(diff) > dist_threshold) || (length(diff_p) > dist_threshold)
            || (length(diff_m) > dist_threshold)) {
          row.result = -4;
          continue;
        }
        if ((dot(projectedNormal, referenceNormal) < normal_threshold)){
          row.result = -5;
          continue;
        }
        row.result = 1;

        float inv_cov = 1.0f;
        row.error = inv_cov * dot(referenceNormal, diff);
        ((float3 *) row.J)[0] = inv_cov * referenceNormal;
        ((float3 *) row.J)[1] =
            inv_cov * cross(projectedVertex, referenceNormal);

        float residual_p = dot(referenceNormal, diff_p);
        float residual_m = dot(referenceNormal, diff_m);
        float fin_diff = (residual_p - residual_m)/(2.f*perturbation);
        float ana_J = - row.J[i];

        if (std::isnan(ana_J) || std::isnan(fin_diff)){
          row.result = -5;
          std::cout<<row.J[i]<<" "<<fin_diff <<std::endl;
          continue;
        }

        float ana_num_diff = fabsf(ana_J - fin_diff);
        sum_diff += ana_num_diff;

        if (ana_num_diff > 0.2){
          std::cout<<row.J[i]<< " "<<fin_diff<<std::endl;
        }
        evaluated_pixels++;
      }
    }
    TOCK("trackKernel", inSize.x * inSize.y);
    float RMSE_percent = fabsf(sum_diff)/evaluated_pixels;

    if (std::isnan(RMSE_percent)) {
      exit(0);
    }
    std::cout<<"error: "<<RMSE_percent<<""<<std::endl;
  }
}



void track_Obj_ICP_Kernel(TrackData *output,
                          const uint2 jacobian_size,
                          const float3 *c2_vertice_live,
                          const float3 *c2_normals_live,
                          uint2 inSize,
                          const float3 *c1_vertice_render,
                          const float3 *c1_Normals_render,
                          uint2 refSize,
                          const Matrix4 T_c2_o2, //to be estimated
                          const Matrix4 T_c1_o1,
                          const Matrix4 T_w_o1,
                          const float4 k,
                          const float dist_threshold,
                          const float normal_threshold,
                          const float3 *) {
  TICK();
  uint2 pixel = make_uint2(0, 0);
  unsigned int pixely, pixelx;
//#pragma omp parallel for \
//      shared(output), private(pixel, pixelx, pixely)
  float perturbation = 1e-5f;
  for(int i = 0; i < 6 ; ++i) {
    float perturbed[6] = {0.f, 0.f, 0.f, 0.f,  0.f, 0.f};
    perturbed[i] = perturbation;
    float evaluated_pixels = 0;
    float sum_diff = 0.f;

    for (pixely = 0; pixely < inSize.y; pixely++) {
      for (pixelx = 0; pixelx < inSize.x; pixelx++) {

        pixel.x = pixelx;
        pixel.y = pixely;
        const unsigned idx = pixel.x + pixel.y * jacobian_size.x;
        TrackData &row = output[idx];

        if (c2_normals_live[pixel.x + pixel.y * inSize.x].x == INVALID) {
          row.result = -1;
          continue;
        }


        const Matrix4 Tplus = toMatrix4(TooN::SE3<>(TooN::Vector<6, const
        float, TooN::Reference>(perturbed))) * T_c2_o2;
        const Matrix4 Tminus = toMatrix4(TooN::SE3<>(-1.f * TooN::Vector<6,
                                                                         const float, TooN::Reference>(perturbed))) * T_c2_o2;

        const float3 c2_vertex = c2_vertice_live[pixel.x + pixel.y * inSize.x];
        float3 o2_vertex, c1_vertex_o1;
        float3 o2_vertex_plus, c1_vertex_o1_plus;
        float3 o2_vertex_minus, c1_vertex_o1_minus;
        const float2 projpixel = obj_warp(c1_vertex_o1, o2_vertex, T_c1_o1,
                                          T_c2_o2, getCameraMatrix(k), c2_vertex);
        const float2 projpixel_p = obj_warp(c1_vertex_o1_plus, o2_vertex_plus, T_c1_o1,
                                            Tplus, getCameraMatrix(k), c2_vertex);
        const float2 projpixel_m = obj_warp(c1_vertex_o1_minus, o2_vertex_minus, T_c1_o1,
                                            Tminus, getCameraMatrix(k), c2_vertex);

        if (projpixel.x < 1  || projpixel.x > inSize.x - 2 || projpixel.y < 1
            || projpixel.y > inSize.y - 2 || projpixel_p.x < 1
            || projpixel_p.x > inSize.x - 2 || projpixel_p.y < 1
            || projpixel_p.y > inSize.y - 2
            || projpixel_m.x < 1
            || projpixel_m.x > inSize.x - 2 || projpixel_m.y < 1
            || projpixel_m.y > inSize.y - 2
            ) {
          row.result = -2;
          continue;
        }

        const float3 c_normal = c2_normals_live[pixel.x + pixel.y * inSize.x];
        const float3 o2_normal = rotate(inverse(T_c2_o2), c_normal);
//      const float3 c1_normal_o1 = rotate(inverse(T_c1_o1), o2_normal);

//      const float3 projectedPos = getCameraMatrix(k) * c1_vertex_o1;
//      const float2 projPixel = make_float2(
//          projectedPos.x / projectedPos.z + 0.5f,
//          projectedPos.y / projectedPos.z + 0.5f);
//      if (projPixel.x < 0 || projPixel.x > refSize.x - 1
//          || projPixel.y < 0 || projPixel.y > refSize.y - 1) {
//        row.result = -2;
//        continue;
//      }

        /*
        const float3 projectedPos_plus = getCameraMatrix(k) * c0_vertex_0_plus;
        const float2 projPixel_plus = make_float2(
            projectedPos_plus.x / projectedPos_plus.z + 0.5f,
            projectedPos_plus.y / projectedPos_plus.z + 0.5f);
  */
        float residual, residual_p, residual_m;
        float3 o1_refNormal_o1, o1_refNormal_o1_p, o1_refNormal_o1_m;
        float3 diff, diff_p, diff_m;
        bool has_residual = obj_icp_residual(residual, o1_refNormal_o1, diff,
                                             T_w_o1, o2_vertex, c1_vertice_render, c1_Normals_render, inSize, projpixel);
        bool has_residual_p = obj_icp_residual(residual_p, o1_refNormal_o1_p, diff_p,
                                               T_w_o1, o2_vertex_plus, c1_vertice_render, c1_Normals_render, inSize, projpixel);
        bool has_residual_m = obj_icp_residual(residual_m, o1_refNormal_o1_m, diff_m,
                                               T_w_o1, o2_vertex_minus, c1_vertice_render, c1_Normals_render, inSize, projpixel);
//      const uint2 refPixel = make_uint2(projPixel.x, projPixel.y);
//      const float3 w_refNormal_o1 = c1_Normals_render[refPixel.x + refPixel.y * refSize.x];
//      const float3 w_refVertex_o1 = c1_vertice_render[refPixel.x + refPixel.y * refSize.x];
//      if (w_refNormal_o1.x == INVALID) {
//        row.result = -3;
//        continue;
//      }
        if ((!has_residual_m) || (!has_residual_p) || (!has_residual)) {
          row.result = -3;
          continue;
        }

        if (length(diff) > dist_threshold) {
          row.result = -4;
          continue;
        }
        if (dot(o2_normal, o1_refNormal_o1) < normal_threshold) {
          row.result = -5;
          continue;
        }

        //calculate the inverse of covariance as weights
//      const float3 P = icp_cov_layer[pixel.x + pixel.y * inSize.x];
//      const float sigma_icp = o_reformal.x * o_reformal.x * P.x
//          + o_reformal.y * o_reformal.y * P.y
//          + o_reformal.z * o_reformal.z * P.z;
//      const float inv_cov = sqrtf(1.0 / sigma_icp);

        const float inv_cov = 1.0f;
//      row.error = inv_cov * dot(o1_refNormal_o1, diff);
//      std::cout<<row.error<<std::endl;

//      float3 Jtrans = rotate(o1_refNormal_o1, transpose(T_c2_o2));
        float3 Jtrans = rotate(T_c2_o2, o1_refNormal_o1);
        ((float3 *) row.J)[0] = -1.0f * inv_cov * - 1.0f * Jtrans;
        ((float3 *) row.J)[1] = /*-1.0f */ inv_cov * cross(c2_vertex, Jtrans);

        float fin_diff = (residual_p - residual_m)/(2.f*perturbation);
        float ana_J = - row.J[i];

        if (std::isnan(ana_J) || std::isnan(fin_diff)){
          row.result = -5;
          std::cout<<row.J[i]<<" "<<fin_diff <<std::endl;
          continue;
        }
        float ana_num_diff = fabsf(ana_J - fin_diff);
        sum_diff += ana_num_diff;

        if (ana_num_diff > 0.2){
          std::cout<<row.J[i]<< " "<<fin_diff<<std::endl;
        }
        evaluated_pixels++;
        row.result = 1;
      }
    }

    TOCK("trackKernel", inSize.x * inSize.y);
    float RMSE_percent = fabsf(sum_diff)/evaluated_pixels;

    if (std::isnan(RMSE_percent)) {
      exit(0);
    }
    std::cout<<"error: "<<RMSE_percent<<""<<std::endl;
  }
}



void Tracking::track_obj_RGB_kernel(TrackData* output, const uint2 jacobian_size,
                                    const float3 *r_vertices_render,
                                    const float3 *r_vertices_live, const float* r_image,
                                    uint2 r_size, const float* l_image, uint2 l_size,
                                    const float * l_gradx, const float * l_grady,
                                    const Matrix4& T_c2_o2, //to be estimated
                                    const Matrix4& T_c1_o1,
                                    const Matrix4& K, const float residual_criteria,
                                    const float grad_threshold, const float sigma_bright) {
  TICK();
  uint2 r_pixel = make_uint2(0, 0);
  unsigned int r_pixely, r_pixelx;
//#pragma omp parallel for shared(output),private(r_pixel,r_pixelx,r_pixely)
  float perturbation = 1e-5f;
  for(int i = 0; i < 6 ; ++i) {
    float perturbed[6] = {0.f, 0.f, 0.f, 0.f,  0.f, 0.f};
    perturbed[i] = perturbation;
    float evaluated_pixels = 0;
    float sum_diff = 0.f;
    for (r_pixely = 0; r_pixely < r_size.y; r_pixely++) {
      for (r_pixelx = 0; r_pixelx < r_size.x; r_pixelx++) {
        r_pixel.x = r_pixelx;
        r_pixel.y = r_pixely;

        TrackData & row = output[r_pixel.x + r_pixel.y * jacobian_size.x];
        const int r_index = r_pixel.x + r_pixel.y * r_size.x;
        float3 r_vertex_render = r_vertices_render[r_index];
        const float3 r_vertex_live = r_vertices_live[r_index];

        //if rendered depth is not available
        if ((r_vertex_render.z <= 0.f) || (r_vertex_render.z == INVALID)) {
          //if live depth is not availvle too =>depth error
//        if (r_vertex_live.z <= 0.f ||r_vertex_live.z == INVALID) {
          row.result = -1;
          continue;
//        }

          /*else{
  //          if live depth is availvle, use live depth instead
  //          would introduce occlusion however
            r_vertex_render = r_vertex_live;
          }*/
        }

//      //if the difference between rendered and live depth is too large =>occlude
//      if (length(r_vertex_render - r_vertex_live) > occluded_depth_diff_){
//        //not in the case that no live depth
//        if (r_vertex_live.z > 0.f){
//          row.result = -3;
////          std::cout<<r_vertex_render.z <<" "<<r_vertex_live.z<<std::endl;
//          continue;
//        }
//      }

        const Matrix4 Tplus = toMatrix4(TooN::SE3<>(TooN::Vector<6, const float, TooN::Reference>(perturbed))) * T_c2_o2;
        const Matrix4 Tminus = toMatrix4(TooN::SE3<>(-1.f * TooN::Vector<6, const float, TooN::Reference>(perturbed))) * T_c2_o2;

        float3 o1_vertex, c2_vertex_o1;
        float3 o1_vertex_plus, c2_vertex_o1_plus;
        float3 o1_vertex_minus, c2_vertex_o1_minus;
        const float2 projpixel = obj_warp(c2_vertex_o1, o1_vertex,
                                          T_c2_o2, T_c1_o1, K, r_vertex_render);
        const float2 projpixel_p = obj_warp(c2_vertex_o1_plus, o1_vertex_plus,
                                            Tplus, T_c1_o1, K, r_vertex_render);
        const float2 projpixel_m = obj_warp(c2_vertex_o1_minus, o1_vertex_minus,
                                            Tminus, T_c1_o1, K, r_vertex_render);

        if (projpixel.x < 1  || projpixel.x > l_size.x - 2 || projpixel.y < 1
            || projpixel.y > l_size.y - 2
            || projpixel_p.x < 1 || projpixel_p.x > l_size.x - 2 || projpixel_p.y < 1
            || projpixel_p.y > l_size.y - 2
            || projpixel_m.x < 1
            || projpixel_m.x > l_size.x - 2 || projpixel_m.y < 1
            || projpixel_m.y > l_size.y - 2
            ) {
          row.result = -2;
          continue;
        }

        const float residual = rgb_residual(r_image, r_pixel, r_size, l_image, projpixel, l_size);
        const float residual_p = rgb_residual(r_image, r_pixel, r_size, l_image, projpixel_p, l_size);
        const float residual_m = rgb_residual(r_image, r_pixel, r_size, l_image, projpixel_m, l_size);

//      const float inv_cov = 1.0/sigma_bright;
        const float inv_cov = 1.0f;
        bool gradValid = obj_rgb_jacobian(row.J, c2_vertex_o1, projpixel,
                                          l_gradx, l_grady, l_size, K, grad_threshold, inv_cov);
        //threshold small gradients
//      if (gradValid == false) {
//        row.result = -5;
//        continue;
//      }
        row.error = inv_cov * residual;

//      if (row.error  * row.error > residual_criteria){
////        std::cout<<row.error<<std::endl;
//        row.result = -4;
//        continue;
//      }

        float fin_diff = (residual_p - residual_m)/(2.f*perturbation);
        float ana_J = - row.J[i];

        if (std::isnan(ana_J) || std::isnan(fin_diff)){
          row.result = -5;
          std::cout<<row.J[i]<<" "<<fin_diff <<std::endl;
          continue;
        }
//
        float ana_num_diff = fabsf(ana_J - fin_diff);
        sum_diff += ana_num_diff;

        if (ana_num_diff > 0.1){
          std::cout<<row.J[i]<< " "<<fin_diff<<std::endl;
        }
        evaluated_pixels++;
        row.result = 1;
      }
    }
    TOCK("trackKernel", inSize.x * inSize.y);
    float RMSE_percent = fabsf(sum_diff)/evaluated_pixels;

    if (std::isnan(RMSE_percent)) {
      exit(0);
    }
    std::cout<<"error: "<<RMSE_percent<<""<<std::endl;
    std::cout<<"evaluated pixels: "<<evaluated_pixels<<std::endl;
  }
}


TEST_F(JacobianTest, FullTrackingTest) {

  TrackData** J = (TrackData **) calloc(sizeof(TrackData*) * pyramid_levels, 1);
  for (uint i = 0; i<pyramid_levels; ++i){
    J[i] = (TrackData *)calloc(sizeof(TrackData) * framesize.x * framesize.y, 1);
  }

//  const float rgb_tracking_threshold_[3] = {0.1, 0.1, 0.01};
  const int num_pyramid = 3;




  for (unsigned int i = 0; i < pyramid_levels; ++i) {
    scaled_r_depth[i] = (float*) calloc(
        sizeof(float) * (framesize.x * framesize.y)
            / (int) pow(2, i), 1);
    scaled_l_image[i] = (float*) calloc(
        sizeof(float) * (framesize.x * framesize.y)
            / (int) pow(2, i), 1);
    scaled_r_image[i] = (float*) calloc(
        sizeof(float) * (framesize.x * framesize.y)
            / (int) pow(2, i), 1);
    scaled_l_gradx[i] = (float*) calloc(
        sizeof(float) * (framesize.x * framesize.y)
            / (int) pow(2, i), 1);
    scaled_l_grady[i] = (float*) calloc(
        sizeof(float) * (framesize.x * framesize.y)
            / (int) pow(2, i), 1);
    scaled_r_vertex[i] = (float3*) calloc(
        sizeof(float3) * (framesize.x * framesize.y)
            / (int) pow(2, i), 1);
  }


//  memcpy(scaled_depth[0], depthf, sizeof(float) * framesize.x * framesize.y);

  // ********* BEGIN : Generate the gaussian *************
  float * gaussian;
  size_t gaussianS = radius * 2 + 1;
  gaussian = (float*) calloc(gaussianS * sizeof(float), 1);
  int x;
  for (unsigned int i = 0; i < gaussianS; i++) {
    x = i - 2;
    gaussian[i] = expf(-(x * x) / (2 * delta * delta));
  }
  // ********* END : Generate the gaussian *************
  bilateralFilterKernel(scaled_r_depth[0], r_depthf, framesize, gaussian, 0.1f, 2);
  bilateralFilterKernel(scaled_r_image[0], r_image, framesize, gaussian, 0.1f, 2);
  bilateralFilterKernel(scaled_l_image[0], l_image, framesize, gaussian, 0.1f, 2);

//  memcpy(scaled_right[0], right, sizeof(float) * framesize.x * framesize.y);
//  memcpy(scaled_left[0], left, sizeof(float) * framesize.x * framesize.y);

////  convert opencv mat to array
//  //for debuging
////  cv::Mat im2=cv::imread("/home/binbin/code/octree-lib-wip/apps/kfusion/test/tracking/720.png", 0);
////  cv::Mat im1=cv::imread("/home/binbin/code/octree-lib-wip/apps/kfusion/test/tracking/721.png", 0);
////  cv::Mat depth2=cv::imread("/home/binbin/code/octree-lib-wip/apps/kfusion/test/tracking/721_depth.png", 0);
//
//  cv::Mat im2color=cv::imread("/home/binbin/code/octree-lib-wip/apps/kfusion/test/tracking/tum1.png");
//  cv::Mat im1color=cv::imread("/home/binbin/code/octree-lib-wip/apps/kfusion/test/tracking/tum2.png");
//  cv::Mat depth2=cv::imread("/home/binbin/code/octree-lib-wip/apps/kfusion/test/tracking/tum2-depth.png", -1);
//  if (im2color.empty() || depth2.empty() || im1color.empty()) {
//    std::cerr << "Not correct RGB-D images";
//  }
//
//  cv::Mat im1Grey, im2Grey, depth2f; /*in meters*/;
//  cv::cvtColor(im1color, im1Grey, CV_BGR2GRAY);
//  cv::cvtColor(im2color, im2Grey, CV_BGR2GRAY);
//  depth2.convertTo(depth2f, CV_32FC1, 0.0002);
//
//  std::vector<cv::Mat> im1Pyramid, im2Pyramid, depthPyramid, gradXPyramid, gradYPyramid;
//
//  cv::Mat1f im1f, im2f;
//  im1Grey.convertTo(im1f, CV_32FC1, 1/255.0f);
//  im2Grey.convertTo(im2f, CV_32FC1, 1/255.0f);
//
//  im1Pyramid.push_back(im1f);
//  im2Pyramid.push_back(im2f);
//  depthPyramid.push_back(depth2f);
//
//  std::memcpy(scaled_right[0], im1Pyramid[0].data, sizeof(float) * framesize.x * framesize.y);
//  std::memcpy(scaled_left[0], im2Pyramid[0].data, sizeof(float) * framesize.x * framesize.y);
//  std::memcpy(scaled_depth[0], depthPyramid[0].data, sizeof(float) * framesize.x * framesize.y);
//
//  cv::Mat im1PyrUp = im1f;
//  cv::Mat im2PyrUp = im2f;
//  cv::Mat depthPyrUp = depth2f;
//
//  cv::Mat gradx, grady;
//  const int sobelWindow = 5;
//  cv::Sobel(im1f, gradx, CV_32F, 1, 0, sobelWindow);
//  cv::Sobel(im2f, grady, CV_32F, 0, 1, sobelWindow);
//
//  gradXPyramid.push_back(gradx);
//  gradYPyramid.push_back(grady);
//  std::memcpy(scaled_gradx[0], gradXPyramid[0].data, sizeof(float) * framesize.x * framesize.y);
//  std::memcpy(scaled_grady[0], gradYPyramid[0].data, sizeof(float) * framesize.x * framesize.y);

  // half sample the input depth maps into the pyramid levels
  for (unsigned int i = 1; i < pyramid_levels; ++i) {

    halfSampleRobustImageKernel(scaled_r_depth[i], scaled_r_depth[i - 1],
                                make_uint2(framesize.x / (int) pow(2, i - 1),
                                           framesize.y / (int) pow(2, i - 1)), e_delta * 3, 1);

    halfSampleRobustImageKernel(scaled_l_image[i], scaled_l_image[i - 1],
                                make_uint2(framesize.x / (int) pow(2, i - 1),
                                           framesize.y / (int) pow(2, i - 1)), e_delta * 3, 1);

    halfSampleRobustImageKernel(scaled_r_image[i], scaled_r_image[i - 1],
                                make_uint2(framesize.x / (int) pow(2, i - 1),
                                           framesize.y / (int) pow(2, i - 1)), e_delta * 3, 1);
////
//      GaussianDownsamplingKernel(scaled_depth[i], scaled_depth[i - 1],
//                                 make_uint2(framesize.x / (int) pow(2, i - 1),
//                                            framesize.y / (int) pow(2, i - 1)));
//      GaussianDownsamplingKernel(scaled_left[i], scaled_left[i - 1],
//                                 make_uint2(framesize.x / (int) pow(2, i - 1),
//                                            framesize.y / (int) pow(2, i - 1)));
//      GaussianDownsamplingKernel(scaled_right[i], scaled_right[i - 1],
//                                 make_uint2(framesize.x / (int) pow(2, i - 1),
//                                            framesize.y / (int) pow(2, i - 1)));
////
////      opencv
//      cv::Mat im1PyrDown, im2PyrDown, depthDown;
//      cv::pyrDown(im1PyrUp, im1PyrDown, cv::Size( im1PyrDown.cols/2, im1PyrDown.rows/2 ));
//      cv::pyrDown(im2PyrUp, im2PyrDown, cv::Size( im2PyrDown.cols/2, im2PyrDown.rows/2 ));
//      cv::pyrDown(depthPyrUp, depthDown, cv::Size( depthDown.cols/2, depthDown.rows/2 ));
//
//      im1Pyramid.push_back(im1PyrDown);
//      im2Pyramid.push_back(im2PyrDown);
//      depthPyramid.push_back(depthDown);
//
//      int downsampledSize = framesize.x / (int) pow(2, i) * framesize.y / (int) pow(2, i);
//
//      std::memcpy(scaled_right[i], im1Pyramid[i].data, sizeof(float) * downsampledSize);
//      std::memcpy(scaled_left[i], im2Pyramid[i].data, sizeof(float) * downsampledSize);
//      std::memcpy(scaled_depth[i], depthPyramid[i].data, sizeof(float) * downsampledSize);
//
//      im1PyrDown.copyTo(im1PyrUp);
//      im2PyrDown.copyTo(im2PyrUp);
//      depthDown.copyTo(depthPyrUp);
//      std::cout << "Break Here" << std::endl;
//
////      im1PyrDown.convertTo(im1PyrDown, CV_8UC1, 255.0f);
//      cv::Sobel(im1Pyramid[i], gradx, CV_64F, 1, 0, sobelWindow);
//      cv::Sobel(im1Pyramid[i], grady, CV_64F, 0, 1, sobelWindow);
//
//      gradXPyramid.push_back(gradx);
//      gradYPyramid.push_back(grady);
//      std::memcpy(scaled_gradx[i], gradXPyramid[i].data, sizeof(float) * downsampledSize);
//      std::memcpy(scaled_grady[i], gradYPyramid[i].data, sizeof(float) * downsampledSize);
//
//      std::ostringstream name;
//      name <<"./results"<<"/pyramid_levels_"<<i<<"_channel_"<<im1PyrDown.channels()<<".png";
//      im1PyrDown.convertTo(im1PyrDown, CV_8UC1);
//      cv::imwrite(name.str(), im1PyrDown);
  }

  for (unsigned int i = 0; i < pyramid_levels; ++i) {
    uint2 localimagesize = make_uint2(
        framesize.x / (int) pow(2, i),
        framesize.y / (int) pow(2, i));
    Matrix4 invK = getInverseCameraMatrix(k / float(1 << i));
    depth2vertexKernel(scaled_r_vertex[i], scaled_r_depth[i], localimagesize, invK);
    gradientsKernel(scaled_l_gradx[i], scaled_l_grady[i], scaled_l_image[i],
                    localimagesize);
//      SobelGradientsKernel(scaled_gradx[i], scaled_grady[i], scaled_right[i], localimagesize, 3);
  }

  Matrix4 T_w_l = GT_r;
  Matrix4 T_w_r = GT_r;

  std::vector<float> errors;


  for(int level = pyramid_levels - 1; level >= 0; --level) {
    float previous_error = INFINITY;
    uint2 localimagesize = make_uint2(
        framesize.x / (int) pow(2, level),
        framesize.y / (int) pow(2, level));
    K = getCameraMatrix(k / float(1 << level));
    for(int i = 0; i < pyramid_iter[level]; ++i){
      Matrix4 pose_update = Identity();
      rgbTracker->trackRGB(J[level], framesize, scaled_r_vertex[level],
                           scaled_r_image[level], localimagesize, scaled_l_image[level],
                           localimagesize, scaled_l_gradx[level], scaled_l_grady[level],
                           T_w_r, T_w_l, getCameraMatrix(k / (1 << level)),
                           rgb_tracking_threshold_[level],
                           mini_gradient_magintude[level], 4.0/255.0f);

      float * reductionoutput = (float*) calloc(sizeof(float) * 8 * 32, 1);
      rgbTracker->reduceKernel(reductionoutput,
                               J[level],
                               framesize, localimagesize);

      const float current_error = reductionoutput[0]/reductionoutput[28];
      std::cout<< "Level " << level << ", Iteration " << i << ", Error: " << current_error<<std::endl;

      if (current_error > (previous_error)){
        pose_update = Identity();
        std::cout<< "Level " << level << " Iteration " << i << " - Cost increase from " << previous_error << " to " <<
                 current_error<<std::endl;
        break;
      }
      previous_error = current_error;

      if(rgbTracker->solvePoseKernel(pose_update, reductionoutput, 1e-5)) {
        T_w_l = pose_update * T_w_l;
        std::cout<<"pose updating done! Break!"<<std::endl;
        break;
      }

      T_w_l = pose_update * T_w_l;
      printMatrix4("updated live pose", T_w_l);
      // std::cout << pose_update << std::endl;
      // std::cout << pose << std::endl;
      errors.push_back(current_error);

      cv::Mat errorImg;
      renderError(errorImg, J[level], localimagesize, framesize);
      std::ostringstream name;
      name <<"./results"<<"/level_"<<level<<"_iteration_"<<i<<".png";
      cv::imwrite(name.str(), errorImg);

      cv::Mat warpedImg;
      cv::Mat residualImg;
      rgbTracker->warp_to_residual(warpedImg, residualImg, l_image, r_image,
                                   r_vertex, T_w_l, T_w_r, getCameraMatrix(k / (1 << level)), localimagesize);

//      std::ostringstream name_warp;
//      name_warp <<"./results"<<"/level_"<<level<<"_iteration_"<<i<<"_warped"<<".png";
//      cv::imwrite(name_warp.str(), warpedImg);
//
//      std::ostringstream name_residual;
//      name_residual <<"./results"<<"/level_"<<level<<"_iteration_"<<i<<"_residual"<<".png";
//      cv::imwrite(name_residual.str(), residualImg);

//      cv::Mat inImage;
//      renderImage(inImage, scaled_r_image[level], localimagesize);
//      std::ostringstream name_input;
//      name_input <<"./results"<<"/level_"<<level<<"_iteration_"<<i<<"_ref"<<".png";
//      cv::imwrite(name_input.str(), inImage);

//      cv::Mat depthImage;
//      renderImage(depthImage, scaled_r_depth[i], localimagesize);
////      cv::namedWindow("error image", CV_WINDOW_AUTOSIZE );
//      std::ostringstream name_depth;
//      name_depth <<"./results"<<"/level_"<<level<<"_iteration_"<<i<<"_depth"<<".png";
//      cv::imwrite(name_depth.str(), depthImage);

//      cv::imshow("error image", errorImg );
//      cv::Mat visualError;
//      cv::applyColorMap(errorImg, visualError, cv::COLORMAP_PARULA);
//      cv::imwrite(name.str(), visualError);
    }
  }

  Matrix4 T_l_r = inverse(T_w_l) * T_w_r;
  Matrix4 GT_l_r = inverse(GT_l) * GT_r;

  std::cout << "Computed: \n" << T_l_r  << std::endl;
  std::cout << "GT: \n" << GT_l_r << std::endl;
//  std::cout << "GT_rotation: \n" << GT_pose << std::endl;
//  float trans_diff = 0.0f;
  const float3 estimated_trans = get_translation(T_l_r);
  const float3 GT_trans = get_translation(GT_l_r);
  const float trans_est_mag = length(estimated_trans);
  const float trans_GT_mag = length(GT_trans);
  const float trans_diff_mag = length(estimated_trans - GT_trans);
  std::cout<< "Estimated translation: \n"<<trans_est_mag * 1000.0f<<" "
                                                                    "[mm]"<<std::endl;
  std::cout<< "Ground truth translation: \n"<<trans_GT_mag * 1000.0f<<" "
                                                                      "[mm]"<<std::endl;
  std::cout<< "Error in translation drift: \n"<<trans_diff_mag * 1000.0f<<" "
                                                                          "[mm] => around "<<trans_diff_mag/trans_GT_mag * 100.0f<<"%"<<std::endl;
  ASSERT_NEAR(errors[errors.size()-1], 0.f, 1e-4);
}

void dump_residual(TrackData* res_data, const uint2 inSize){
  std::ofstream residual_file;
  residual_file.open("residual.csv");
  unsigned int pixely, pixelx;
  for (pixely = 0; pixely < inSize.y; pixely++) {
    for (pixelx = 0; pixelx < inSize.x; pixelx++) {
      int id = pixelx + pixely * inSize.x;
      if (res_data[id].result < 1) continue;
      residual_file<<(res_data[id].error)<<",\n";
    }
  }
  residual_file.close();
}

TEST_F(JacobianTest, CallClassFunctionTEST) {
  Matrix4 T_w_l;
  rgbTracker->robustWeight = RobustW::noweight;

  rgbTracker->set_params_frame(k, l_image, l_depthf);
  rgbTracker->buildPreviousPyramid(r_image, r_depthf);

  bool tracked = rgbTracker->trackLiveFrame(T_w_l, GT_r, k, r_vertex,
                                            r_vertex, l_image, l_depthf);
  dump_residual(rgbTracker->getTrackingResult(), framesize);

  Matrix4 T_l_r = inverse(T_w_l) * GT_r;
  Matrix4 GT_l_r = inverse(GT_l) * GT_r;

  std::cout << "Computed: \n" << T_l_r  << std::endl;
  std::cout << "GT: \n" << GT_l_r << std::endl;
//  std::cout << "GT_rotation: \n" << GT_pose << std::endl;
//  float trans_diff = 0.0f;
  const float3 estimated_trans = get_translation(T_l_r);
  const float3 GT_trans = get_translation(GT_l_r);
  const float trans_est_mag = length(estimated_trans);
  const float trans_GT_mag = length(GT_trans);
  const float trans_diff_mag = length(estimated_trans - GT_trans);
  std::cout<< "Estimated translation: \n"<<trans_est_mag * 1000.0f<<" "
                                                                    "[mm]"<<std::endl;
  std::cout<< "Ground truth translation: \n"<<trans_GT_mag * 1000.0f<<" "
                                                                      "[mm]"<<std::endl;
  std::cout<< "Error in translation drift: \n"<<trans_diff_mag * 1000.0f<<" "
                                                                          "[mm] => around "<<trans_diff_mag/trans_GT_mag * 100.0f<<"%"<<std::endl;
  EXPECT_LE(trans_diff_mag, 1e-3);
}