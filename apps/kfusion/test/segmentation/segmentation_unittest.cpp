//
// Created by binbin on 19/06/18.
//
#include "math_utils.h"
#include "commons.h"
#include "gtest/gtest.h"

#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>

#include "../../src/cpp/segmentation/segmentation.h"

class SegmentationTest : public ::testing::Test{
 protected:
  std::shared_ptr<Segmentation> segmenter;
  virtual void SetUp (){

    I_r = cv::imread(FILEPATH"/1341846436.889894.png");
    D_r = cv::imread(FILEPATH"/1341846436.889927.png");
    I_l = cv::imread(FILEPATH"/1341846436.922139.png");
    D_l = cv::imread(FILEPATH"/1341846436.922153.png");

    //TUM fr3walking-halfsphere data
    T_w_r = toMatrix4({0.0068, -2.7651, 1.6077, -0.7091}, {-0.2017, 0.2146, 0.6407});
    T_w_l = toMatrix4({0.0167, -2.7562, 1.6066, -0.7062}, {-0.2011, 0.2219, 0.6416});
    k = {525.0, 525.0, 319.5, 239.5};//tum-ros

    //read maskfusion result
    mask_r = FILEPATH"/mask/0064.npy";
    mask_l = FILEPATH"/mask/0065.npy";
    classid_r = FILEPATH"/classid/0064.npy";
    classid_l = FILEPATH"/classid/0065.npy";

    SegmentationResult maskrcnn_r=  segmenter->load_mask_rcnn(classid_r, mask_r,
        framesize.y, framesize.x);
    SegmentationResult maskrcnn_l=  segmenter->load_mask_rcnn(classid_l, mask_l,
        framesize.y, framesize.x);

  }
  Matrix4 T_w_r;
  Matrix4 T_w_l;
  float4 k;

  cv::Mat I_r;
  cv::Mat I_l;
  cv::Mat D_r;
  cv::Mat D_l;

  uint2 framesize = {640, 480};

  std::string mask_r;
  std::string mask_l;
  std::string classid_r;
  std::string classid_l;


};

TEST_F(SegmentationTest, realtimeMaskRCNN){
  bool run = true;
  ASSERT_EQ(run, 1);
}


