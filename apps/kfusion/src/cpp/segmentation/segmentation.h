 /*
 * SPDX-FileCopyrightText: 2017-2019 Smart Robotics Lab, Imperial College London
 * SPDX-FileCopyrightText: 2017-2019 Binbin Xu
 * SPDX-License-Identifier: BSD-3-Clause
 */


#ifndef SEGMENTATION_H
#define SEGMENTATION_H

#include <dirent.h>
#include <vector>
#include <string>
#include <iostream>
#include <iostream>
#include <cstring>
#include <algorithm>
#include "cnpy.h"
#include <omp.h>
#include <vector_types.h>
#include <commons.h>
#include <math_utils.h>
#include "../preprocessing/preprocessing.h"
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <map>

//# COCO Class names
//# Index of the class in the list is its ID. For example, to get ID of
//# the teddy bear class, use: class_names.index('teddy bear')
//# The index of the class name in the list represent its ID
//# (first class is 0, second is 1, third is 2, ...etc.)
//# in total 91 indexs
//class_names = [
//'BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane','bus', 'train', 'truck',
// 'boat', //'traffic light', fire hydrant', 'stop sign', 'parking meter',
// 'bench', 'bird','cat', 'dog', 'horse',
// 'sheep', 'cow',// 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
// 'umbrella', 'handbag', 'tie',
//'suitcase', frisbee',// 'skis', 'snowboard', 'sports ball','kite',
// 'baseball bat', 'baseball glove', 'skateboard','surfboard',
// 'tennis racket', //40 => 'bottle','wine glass', 'cup','fork', 'knife',
// 'spoon', 'bowl', 'banana', 'apple',
//'sandwich',//50 => 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza','donut',
// 'cake', 'chair', 'couch',
// 'potted plant',//60=> 'bed','dining table', 'toilet', 'tv', 'laptop',
// 'mouse', 'remote','keyboard', 'cell phone',
// 'microwave',//70=> 'oven', 'toaster','sink', 'refrigerator', 'book', 'clock',
// 'vase', 'scissors','teddy bear',
// 'hair drier', 'toothbrush'
//]

struct instance_seg{
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
  int class_id_;
  cv::Mat instance_mask_;
  All_Prob_Vect all_prob_;
  bool rendered_;
  bool hasModelMask_;
  cv::Mat global_mask_; //in case there is a corresponding mask
  int recog_time_;

  instance_seg(const int class_id, const cv::Mat& instance_mask,
               const All_Prob_Vect& all_prob):
      class_id_(class_id),
      instance_mask_(instance_mask),
      all_prob_(all_prob)
  {rendered_ = false;
    hasModelMask_ = false;
    global_mask_ = cv::Mat::zeros(instance_mask.size(), instance_mask.type());
    recog_time_ = 1;}

  instance_seg(const cv::Mat& instance_mask):
      class_id_(-2),//invalid
      instance_mask_(instance_mask),
      all_prob_(All_Prob_Vect::Zero())
  {rendered_ = false;
    hasModelMask_ = false;
    global_mask_ = cv::Mat::zeros(instance_mask.size(), instance_mask.type());
    recog_time_ = 1;}

  instance_seg(const int class_id, const cv::Mat& instance_mask):
      class_id_(class_id),
      instance_mask_(instance_mask),
      all_prob_(All_Prob_Vect::Zero())
  {rendered_ = false;
    hasModelMask_ = false;
    global_mask_ = cv::Mat::zeros(instance_mask.size(), instance_mask.type());
    recog_time_ = 1;
    if (class_id != 255) all_prob_[class_id] = 1.0f;
  }

  ~instance_seg(){};

  ///
  /// Dilate mask to generate background layer
  /// \param generalized output genralized label map
  //label: 1/255 (fg); 0 (bg); -1 (don't fuse)
  /// \param input original (instance-level) label map (binary)
  void generalize_label(cv::Mat& gener_inst_mask) const;

  //merge two instances (same semantic class) together
  void merge(const instance_seg& new_instance);

  void setGlobalMask(const cv::Mat& globalMask){
    global_mask_ = globalMask;
  }

  //background dilation kernel size
  static constexpr uint bg_dilute_size_ = 5;//9
  static constexpr float render_mask_size_ratio_ = 0.03f;
  static constexpr float large_render_fg_ = 0.5f;
  static constexpr float small_render_fg_ = 0.1f;
};

struct SegmentationResult {

  int width_;
  int height_;
  cv::Mat labelImg;
  std::map<int, instance_seg, std::less<int>,
           Eigen::aligned_allocator<std::pair<const int, instance_seg> > >
      pair_instance_seg_;
  uint label_number_;

  void combine_labels();
  void generate_bgLabels();


  SegmentationResult(int width, int height): width_(width), height_(height){
    labelImg = cv::Mat::zeros(cv::Size(height, width), CV_32SC1);
  }

  SegmentationResult(uint2 imgSize){
    width_ = imgSize.y;
    height_ = imgSize.x;
    labelImg = cv::Mat::zeros(cv::Size(imgSize.x, imgSize.y), CV_32SC1);
  }

  void reset(){
    labelImg = cv::Mat::zeros(cv::Size(height_, width_), CV_32SC1);
    pair_instance_seg_.clear();
  }

//  int find_classID_from_instaneID(const int& instance_id, bool info) const;
//  bool find_classProb_from_instID(All_Prob_Vect& all_prob, const int &instance_id,
//                                  bool info) const;
//  instance_seg find_segment_from_instID(const int instance_id, bool info) const;
  void output(const uint frame, const std::string& str ) const ;
  void print_class_all_prob() const;

//  void merge_mask_rcnn(const float overlap_ratio);

  void set_render_label(const bool rendered);
  void exclude_human();
};

class Segmentation{
 private:
  //to be tuned
 // for co-fusion car dataset
//  const float geom_threshold = 0.01f;
//  const float geometric_lambda = 0.8f;
//  //let's oversegment everything
//  const uint geomtric_component_size = 0;
//  const uint geom_dilute_size = 13;

//  //for tum dataset
//  static constexpr float geom_threshold = 0.008f; //smoothenss, edgeness
//  static constexpr float geometric_lambda = 0.20f;

  //qualitative evaluations
  static constexpr float geom_threshold = 0.001f; //smoothenss(大), edgeness
      // (小)//0.001f  // 0.0005f;
  static constexpr float geometric_lambda = 0.05f; //0.05f //0.001f

//let's oversegment everything
  static constexpr uint geomtric_component_size = 0;
  const std::set<int> filter_class_id_ = {60 /*bed*/,61 /*dinning
 * table*/, 73/*refrigrator*/, 62 /*toilet*/};
  const std::set<int> human_classid_ = {1, 255};

 private:
//  Segmentation();
//loading mask rcnn
  void load_mask_kernel(bool * mask_pixel, cnpy::NpyArray mask_npy, int class_id);

//the format of class id would be 1D interger, with the element of the class id
  std::vector<int> load_class_ids(std::string class_path);

  //the format of class probability would be #Object * #Class float
  std::vector<All_Prob_Vect> load_class_prob(const std::string prob_path,
                                      uint&class_num);

//load mask from all frame infomation
  std::vector<cv::Mat> load_mask(std::string mask_path, int width, int height);


  //calculate edges
  float calc_distance(const float3* inVertex, const float3* inNormal, const int id, const int id_i);
  float calc_concavity(const float3* inNormal, const int id, const int id_i);
  void geometric_edge_kernel(bool * isEdge, const float3* inVertex, const float3* inNormal,
                             const uint2 inSize, const float lambda, const float threshold);

//  void label2mask(cv::Mat& mask, const cv::Mat& labeledImage, const int label_id);
  Eigen::MatrixXf readNPY(const std::string& filename);

 public:
  enum class METHOD { MASK_RCNN, GEOMETRIC_EDGE, COMBINED};
  static constexpr float geo2mask_threshold = 0.80f;
  static constexpr float mask2model_threshold = 0.5f; //0.95f in maskfusion
  static constexpr float new_model_threshold = 0.01f;
  bool refine_human_boundary_;
  bool semanticfusion = true;
  uint geom_dilute_size = 7;//9

  public:
  //read maskrcnn npy files from the folder file_dir
  std::vector<std::string> readFiles(std::string file_dir);

  //load mask-rcnn
  SegmentationResult load_mask_rcnn(std::string class_path,
                                    std::string mask_path,
                                    const std::string prob_path,
                                    int width, int height);


  SegmentationResult compute_geom_edges(const float * depth, const uint2 inSize, const float4 k);

  //merge different segmentation results
  void mergeLabels(SegmentationResult& mergedSeg,
                   const SegmentationResult& srcSeg,
                   const SegmentationResult& dstSeg,
                   const float threshold);

  SegmentationResult finalMerge(const SegmentationResult& srcSeg,
                                 const SegmentationResult& dstSeg,
                                 const float threshold);

  //mapping masks to models
  bool local2global(SegmentationResult& mergedSeg,
                    SegmentationResult& newModel,
                    const SegmentationResult& mask,
                    const SegmentationResult& model,
                    const int min_mask_size,
                    const float combine_threshold,
                    const float new_model_threshold);

  void remove_human_and_small(SegmentationResult& output,
                              const SegmentationResult& input,
                              const int min_mask_size);

};



#endif //SEGMENTATION_H
