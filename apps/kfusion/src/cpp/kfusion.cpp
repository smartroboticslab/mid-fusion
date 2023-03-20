/*
 * SPDX-FileCopyrightText: 2016-2019 Emanuele Vespa
 * SPDX-FileCopyrightText: 2017-2019 Smart Robotics Lab, Imperial College London
 * SPDX-FileCopyrightText: 2017-2019 Binbin Xu
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <kernels.h>
#include "timings.h"
#include <perfstats.h>
#include <vtk-io.h>
#include <octree.hpp>
#include "continuous/volume_instance.hpp"
#include "algorithms/meshing.hpp"
#include "geometry/octree_collision.hpp"
#include "preprocessing.h"
#include "rendering/rendering.h"



extern PerfStats Stats;

// input once
float * gaussian;

// inter-frame
//Volume<FieldType> volume;
float3 * vertex;
float3 * normal;



//float3 bbox_min;
//float3 bbox_max;

// intra-frame

/// Coordinate frames:
///   _l:

float * floatDepth;
float3 * inputRGB; //rgb input image
float * g_inputGrey; //grey input image

Matrix4 T_w_r;
Matrix4 raycastPose;


bool bayesian = false;

// For debugging purposes, will be deleted once done.
std::vector<Matrix4> poses;


void Kfusion::languageSpecificConstructor() {

  if (getenv("KERNEL_TIMINGS"))
    print_kernel_timing = true;

  // internal buffers to initialize

  floatDepth = (float*) calloc(
      sizeof(float) * computationSize.x * computationSize.y, 1);
  inputRGB = (float3*) calloc(
      sizeof(float3) * computationSize.x * computationSize.y, 1);
  g_inputGrey = (float*) calloc(
      sizeof(float) * computationSize.x * computationSize.y, 1);
  vertex = (float3*) calloc(
      sizeof(float3) * computationSize.x * computationSize.y, 1);
  normal = (float3*) calloc(
      sizeof(float3) * computationSize.x * computationSize.y, 1);
  vertex_before_fusion = (float3*) calloc(
      sizeof(float3) * computationSize.x * computationSize.y, 1);
  normal_before_fusion = (float3*) calloc(
      sizeof(float3) * computationSize.x * computationSize.y, 1);
//	trackingResult = (TrackData*) calloc(
//			2 * sizeof(TrackData) * computationSize.x * computationSize.y, 1);


  // ********* BEGIN : Generate the gaussian *************
  size_t gaussianS = radius * 2 + 1;
  gaussian = (float*) calloc(gaussianS * sizeof(float), 1);
  int x;
  for (unsigned int i = 0; i < gaussianS; i++) {
    x = i - 2;
    gaussian[i] = expf(-(x * x) / (2 * delta * delta));
  }
  // ********* END : Generate the gaussian *************

  if(config.groundtruth_file != ""){
    parseGTFile(config.groundtruth_file, poses);
    std::cout << "Parsed " << poses.size() << " poses" << std::endl;
  }

  bayesian = config.bayesian;

//  if(frame == segment_startFrame_){
  std::string segFolder = config.maskrcnn_folder;
  std::string maskFolder = segFolder + "/mask/";
  std::string classFolder = segFolder + "/class_id/";
  std::string probFolder = segFolder + "/all_prob/";
  std::vector<std::string> mask_files = segmenter_->readFiles(maskFolder);
  std::vector<std::string> class_files = segmenter_->readFiles(classFolder);
  std::vector<std::string> prob_files = segmenter_->readFiles(probFolder);
  mask_rcnn_result_.reserve(mask_files.size());
  for (size_t i = 0; i < mask_files.size(); ++i) {
    mask_rcnn_result_.push_back(segmenter_->load_mask_rcnn(class_files[i],
                                                           mask_files[i],
                                                           prob_files[i],
                                                           computationSize.y,
                                                           computationSize.x));
  }

  if (config.gt_mask_folder == "") use_GT_segment = false;
  else{
    use_GT_segment = true;
//    GT_label_files_ = segmenter_->readFiles(GT_label_folder);
    GT_mask_files_ = segmenter_->readFiles(config.gt_mask_folder);
  }

//  }
  objects_in_view_.insert(0);
  in_debug_ = config.in_debug;
  if (config.output_images != ""){
    render_output = true;
  }
  else{
    render_output = false;
  }
//  volume.init(volumeResolution.x, volumeDimensions.x);
  min_object_ratio_ = config.min_obj_ratio;
  min_object_size_ = computationSize.x * computationSize.y * min_object_ratio_;
  Object::absorb_outlier_bg = config.absorb_outlier_bg;
  obj_moved_threshold_ = config.obj_moved;
  geom_refine_human = config.geom_refine_human;
  if (geom_refine_human) segmenter_->geom_dilute_size = 12;
  segment_startFrame_ = config.init_frame;
  reset();
}

Kfusion::~Kfusion() {
  free(gaussian);
  free(floatDepth);
  free(g_inputGrey);
  free(inputRGB);
  free(vertex);
  free(normal);
  free(vertex_before_fusion);
  free(normal_before_fusion);
}

void Kfusion::reset() {
}
void init() {
}
;
// stub
void clean() {
}
;

float* Kfusion::getDepth() {
  return floatDepth;
}
bool Kfusion::preprocessing(const ushort * inputDepth, const uint2 inputSize,
                            const bool filterInput){

  mm2metersKernel(floatDepth, computationSize, inputDepth, inputSize);
  if(filterInput){
    bilateralFilterKernel(floatDepth, floatDepth, computationSize, gaussian,
                          e_delta, radius);
  }
  else {
/*
      unsigned int y;
#pragma omp parallel for \
      shared(l_D), private(y)
      for (y = 0; y < computationSize.y; y++) {
        for (unsigned int x = 0; x < computationSize.x; x++) {
          l_D[0][x + y*computationSize.x] = floatDepth[x + y*computationSize.x];
        }
      }
      */
  }
  return true;
}

void Kfusion::save_input(const uchar3 * inputRGB, uint2 inputSize, int frame){
  if (render_output){
    std::string rgb_file;
    rgb_file = config.output_images+"rgb";
    int ratio = inputSize.x / computationSize.x;
    cv::Mat renderImg = cv::Mat::zeros(cv::Size(computationSize.x, computationSize.y), CV_8UC3);
    unsigned int y;
    for (y = 0; y < computationSize.y; y++)
      for (unsigned int x = 0; x < computationSize.x; x++) {
        uint pos = x * ratio + inputSize.x * y * ratio;
        renderImg.at<cv::Vec3b>(y,x)[2] = inputRGB[pos].x; //R
        renderImg.at<cv::Vec3b>(y,x)[1] = inputRGB[pos].y; //G
        renderImg.at<cv::Vec3b>(y,x)[0] = inputRGB[pos].z;  //B
//      std::cout<<data[pos].x<<data[pos].y<<data[pos].z<<std::endl;
      }

    std::ostringstream name;
    name << rgb_file+"_"<<std::setfill('0') << std::setw(5) << std::to_string(frame)<<".png";
    cv::imwrite(name.str(), renderImg);
  }
}

bool Kfusion::preprocessing(const ushort * inputDepth, const uchar3 * _inputRGB,
                            const uint2 inputSize, const bool filterInput){

//  rgb2intensity(I_l[0], computationSize, _inputRGB, inputSize);
  rgb2intensity(g_inputGrey, inputRGB, computationSize, _inputRGB, inputSize);


  mm2metersKernel(floatDepth, computationSize, inputDepth, inputSize);
  if(filterInput){
    bilateralFilterKernel(floatDepth, floatDepth, computationSize, gaussian,
                          e_delta, radius);
  }
  else {
    /*
    unsigned int y;
#pragma omp parallel for \
    shared(l_D), private(y)
    for (y = 0; y < computationSize.y; y++) {
      for (unsigned int x = 0; x < computationSize.x; x++) {
        l_D[0][x + y*computationSize.x] = floatDepth[x + y*computationSize.x];
      }
    }
  */
  }
  return true;
}

bool Kfusion::tracking(float4 k, uint tracking_rate, uint frame) {

  if (frame % tracking_rate != 0)
    return false;

  bool camera_tracked;

  T_w_r = camera_pose; //get old pose T_w_(c-1)

  //camera tracking against all pixels&vertices(excluding human)
  if (human_out_.pair_instance_seg_.find(INVALID)==human_out_.pair_instance_seg_.end()) {
    rgbdTracker_->set_params_frame(k, g_inputGrey, floatDepth);
  }
  else{
    if (in_debug_){
      human_out_.output(frame,"human_out_for_tracking");
    }
    rgbdTracker_->set_params_frame(k, g_inputGrey, floatDepth, human_out_.pair_instance_seg_.at(INVALID).instance_mask_);
  }

  if(!poses.empty()) {
    this->camera_pose = poses[frame];
    this->camera_pose.data[0].w = (this->camera_pose.data[0].w - poses[0].data[0].w) + this->_init_camera_Pose.x;
    this->camera_pose.data[1].w = (this->camera_pose.data[1].w - poses[0].data[1].w) + this->_init_camera_Pose.y;
    this->camera_pose.data[2].w = (this->camera_pose.data[2].w - poses[0].data[2].w) + this->_init_camera_Pose.z;
    printMatrix4("use ground truth pose", camera_pose);
    camera_tracked = true;
  }
  else{

    if (frame == segment_startFrame_){
      //put reference frame information to current frame memory
      rgbdTracker_->setRefImgFromCurr();
      return false;
    }
    //track against all objects except people
    //estimate static/dynamic objects => only track existing&dynamic objects
    camera_tracked = rgbdTracker_->trackLiveFrame(camera_pose,
                                                  T_w_r, k,
//                                                  objectlist_[0]->m_vertex,
//                                                  objectlist_[0]->m_normal);
                                                  vertex, normal);

    objectlist_[0]->virtual_camera_pose_ = camera_pose;

    //for later motion segmentation
    memcpy(objectlist_.at(0)->trackresult_, rgbdTracker_->getTrackingResult(),
           sizeof(TrackData) * computationSize.x * computationSize.y * 2);
    }

  //use ICP&RGB residual to evaluate moving (possibility) for each object

//raycast again to obtain rendered mask and vertices under CURRENT camera
  // pose and LAST object poses.=>for segmentation and motion residuals
  move_obj_set.clear();
  if (raycast_mask_.pair_instance_seg_.size()>1){
    rendered_mask_.reset();
    objects_in_view_.clear();
    raycastObjectList(objectlist_, vertex_before_fusion, normal_before_fusion,
                      rendered_mask_.labelImg, objects_in_view_, camera_pose,
                      k, computationSize, nearPlane,farPlane, config.mu, false);

    split_labels(rendered_mask_, objects_in_view_, objectlist_);
    rendered_mask_.set_render_label(true);
    if (in_debug_){
      rendered_mask_.output(frame, "render_initial_track");
    }

    //calculate ICP and RGB residuals on all vertices in current camera pose
    rgbdTracker_->compute_residuals(camera_pose, T_w_r,
                                    vertex, normal,
                                    vertex_before_fusion,
                                    use_icp_tracking_, use_rgb_tracking_,
                                    residual_threshold_);

    //check mask with tracking result
    //find moving objects
    for (auto object_render= rendered_mask_.pair_instance_seg_.begin();
         object_render != rendered_mask_.pair_instance_seg_.end(); ++object_render){
      int obj_id = object_render->first;
      //skip background
      if (obj_id == 0) continue;

      const cv::Mat& object_mask = object_render->second.instance_mask_;
      cv::Mat object_inlier_mask = object_mask.clone();
      check_static_state(object_inlier_mask, object_mask,
                         rgbdTracker_->getTrackingResult(), computationSize,
                         use_icp_tracking_, use_rgb_tracking_);
      float inlier_ratio = static_cast<float>(cv::countNonZero(object_inlier_mask))
          /(cv::countNonZero(object_mask));
      //connect objectlist with raycastmask
      if ((inlier_ratio<obj_moved_threshold_) &&
          (inlier_ratio>obj_move_out_threshold_) &&
          (Object::static_object.find(object_render->second.class_id_) == Object::static_object.end())){
        objectlist_.at(obj_id)->set_static_state(false);
        move_obj_set.insert(obj_id);
//        std::cout<<"object "<<obj_id<<" moving threshold: "<<inlier_ratio
//                 <<" class id: "<<objectlist_.at(obj_id)->class_id_<<std::endl;
      }
      else{
        objectlist_.at(obj_id)->set_static_state(true);
//        std::cout<<"object "<<obj_id<<" static threshold: "<<inlier_ratio
//                 <<" class id: "<<objectlist_.at(obj_id)->class_id_<<std::endl;
      }

      if (in_debug_){
        std::ostringstream name;
        name << "/home/binbin/code/octree-lib-wip/apps/kfusion/cmake-build-debug/debug/"
             <<"inlier"<<"_frame_"<< frame << "_object_id_" <<obj_id <<".png";
        cv::imwrite(name.str(), object_inlier_mask);

        std::ostringstream name2;
        name2 << "/home/binbin/code/octree-lib-wip/apps/kfusion/cmake-build"
                 "-debug/debug/"
              <<"all"<<"_frame_"<< frame << "_object_id_" <<obj_id <<".png";
        cv::imwrite(name2.str(), object_mask);
      }
    }

    //refine camera motion using all static vertice, if there is moving
    // object
    if ((move_obj_set.size() != 0) &&
        (rendered_mask_.pair_instance_seg_.size() != 0)){
      //build dynamic object mask
      cv::Size imgSize = raycast_mask_.pair_instance_seg_.begin()->second.instance_mask_
          .size();
      cv::Mat dynamic_obj_mask = cv::Mat::zeros(imgSize, CV_8UC1);
      for (auto object_raycast= raycast_mask_.pair_instance_seg_.begin();
           object_raycast != raycast_mask_.pair_instance_seg_.end(); ++object_raycast){
        int obj_id = object_raycast->first;
        if (move_obj_set.find(obj_id) != move_obj_set.end()){
          cv::bitwise_or(object_raycast->second.instance_mask_,
                         dynamic_obj_mask, dynamic_obj_mask);
        }
      }
      if (in_debug_){
        std::ostringstream name;
        name << "/home/binbin/code/octree-lib-wip/apps/kfusion/cmake-build-debug/debug/"
             <<"dynamic_object"<<"_frame_"<< frame <<".png";
        cv::imwrite(name.str(), dynamic_obj_mask);
      }

      //rgbdTracker_->enable_RGB_tracker(false);
      if (human_out_.pair_instance_seg_.find(INVALID)!=human_out_.pair_instance_seg_.end()) {
        cv::bitwise_or(dynamic_obj_mask, human_out_.pair_instance_seg_.at(INVALID).instance_mask_, dynamic_obj_mask);
      }

      camera_tracked = rgbdTracker_->trackLiveFrame(camera_pose,
                                                    T_w_r, k,
                                                    vertex,
                                                    normal, dynamic_obj_mask);
      //rgbdTracker_->enable_RGB_tracker(true);
      objectlist_[0]->virtual_camera_pose_ = camera_pose;
      //for later motion segmentation
      memcpy(objectlist_.at(0)->trackresult_, rgbdTracker_->getTrackingResult(),
             sizeof(TrackData) * computationSize.x * computationSize.y * 2);
    }

    //refine poses of dynamic objects
///  multiple objects tracking
    for (auto exist_object = objectlist_.begin()+1;
         exist_object != objectlist_.end(); ++exist_object){
      if ((*exist_object)->is_static()) {
        //for later motion segmentation
        memcpy((*exist_object)->trackresult_, objectlist_.at(0)->trackresult_,
               sizeof(TrackData) * computationSize.x * computationSize.y * 2);
        continue;
      }
      else{
        rgbdTracker_->trackEachObject(*exist_object, k, camera_pose, T_w_r,
                                      g_inputGrey, floatDepth);
//        (*exist_object)->set_static_state(true);

        memcpy((*exist_object)->trackresult_, rgbdTracker_->getTrackingResult(),
               sizeof(TrackData) * computationSize.x * computationSize.y * 2);
      }
    }
  }

//
////perform motion refinement accordingly
  if (!move_obj_set.empty()){
  //calculate ICP and RGB residuals on all vertices in current camera pose
    rgbdTracker_->compute_residuals(camera_pose, T_w_r,
                                    vertex, normal,
                                    vertex_before_fusion,
                                    use_icp_tracking_, use_rgb_tracking_,
                                    residual_threshold_);
    //for later motion segmentation
    memcpy(objectlist_.at(0)->trackresult_, rgbdTracker_->getTrackingResult(),
           sizeof(TrackData) * computationSize.x * computationSize.y * 2);

    //raycast again to obtain rendered mask and vertices under current camera
    // pose and object poses.=>for segmentation and motion residuals
    rendered_mask_.reset();
    raycastObjectList(objectlist_, vertex_before_fusion, normal_before_fusion,
                      rendered_mask_.labelImg, objects_in_view_, camera_pose,
                      k, computationSize, nearPlane,farPlane, config.mu, false);

    //find the objects on this frame =>only (track and) integrate them
    split_labels(rendered_mask_, objectlist_);
    rendered_mask_.set_render_label(true);
    if (in_debug_){
      rendered_mask_.output(frame, "volume2mask_new");
    }

  }

  if (((rendered_mask_.pair_instance_seg_.size()>1) && (use_rgb_tracking_)) ||
  (!move_obj_set.empty())){

    for (auto exist_object = objectlist_.begin()+1;
         exist_object != objectlist_.end(); ++exist_object){
      if ((*exist_object)->is_static()) {
//        for later motion segmentation
        memcpy((*exist_object)->trackresult_, objectlist_.at(0)->trackresult_,
               sizeof(TrackData) * computationSize.x * computationSize.y * 2);
        continue;
      }
      else{
        rgbdTracker_->compute_residuals((*exist_object)->virtual_camera_pose_, T_w_r,
                                        (*exist_object)->m_vertex, (*exist_object)->m_normal,
                                        (*exist_object)->m_vertex_bef_integ,
                                        use_icp_tracking_, use_rgb_tracking_,
                                        residual_threshold_);

        memcpy((*exist_object)->trackresult_, rgbdTracker_->getTrackingResult(),
               sizeof(TrackData) * computationSize.x * computationSize.y * 2);
      }
    }

  }

    //put reference frame information to current frame memory
  rgbdTracker_->setRefImgFromCurr();

  if (in_debug_){
    printMatrix4("tracking: camera pose", camera_pose);


    for (auto object = objectlist_.begin(); object != objectlist_.end();
         ++object){
      int class_id = (*object)->class_id_;
      std::cout<<"tracking: object id: "<<(*object)->instance_label_
               <<" ,class id: "<<class_id<<std::endl;
      printMatrix4("tracking: object pose ", (*object)->volume_pose_);

    }
  }

  return camera_tracked;
}

//void mask_semantic_output(const SegmentationResult& masks,
//                          const ObjectList& objectlist,
//                          const uint&frame, std::string str ) {
//  for (auto object = objectlist.begin(); object != objectlist.end();
//       ++object) {
//    const int class_id = (*object)->class_id_;
//    const int instance_id = (*object)->instance_label_;
//    cv::Mat labelMask;
////    if (frame>3){//because raycasting starts from frame 2
//    auto object_mask = masks.instance_id_mask.find(class_id);
//    if ( object_mask!= masks.instance_id_mask.end()) {
//      labelMask = object_mask->second;
//      std::ostringstream name;
//      name << "/home/binbin/code/octree-lib-wip/apps/kfusion/cmake-build-debug/debug/"
//           <<str<<"_frame_"<< frame << "_object_id_" <<instance_id
//           <<"_class_id_" <<class_id<<".png";
//      cv::imwrite(name.str(), labelMask);
//    } else {
//      std::cout<<"frame_"<< frame<<"_"<< str<< "_mask missing: " << class_id << std::endl;
//      continue;
//    }
//  }
//}

void mask_instance_output(const SegmentationResult& masks,
                          const ObjectList& objectlist,
                          const uint&frame, std::string str ) {
  for (auto object = objectlist.begin(); object != objectlist.end();
       ++object) {
    const int class_id = (*object)->class_id_;
    const int instance_id = (*object)->instance_label_;
    cv::Mat labelMask;
    //modifify the label mask from segmentation
//    if (frame>3){//because raycasting starts from frame 2 (seems)
    auto object_mask = masks.pair_instance_seg_.find(instance_id);
    if ( object_mask!= masks.pair_instance_seg_.end()) {
      labelMask = object_mask->second.instance_mask_;
      std::ostringstream name;
      name << "/home/binbin/code/octree-lib-wip/apps/kfusion/cmake-build-debug/debug/"
           <<str<<"_frame_"<< frame << "_object_id_" <<instance_id
           <<"_class_id_" <<class_id<<".png";
      cv::imwrite(name.str(), labelMask);
    } else {
      std::cout<<"frame_"<< frame<<"_"<< str<< "_mask missing: " << instance_id << std::endl;
      continue;
    }
  }
}

bool Kfusion::raycasting(float4 k, float mu, uint frame) {

  bool doRaycast = false;

  std::cout<<"objects (id) in this view: ";
  for(const auto& id: objects_in_view_){
    std::cout<<id<<" ";
  }
  std::cout<<std::endl;

//  if(frame > 2) {
    raycastPose = camera_pose;
//    single frame raycasting
//    raycastKernel(static_map_->volume_, vertex, normal, computationSize,
//        raycastPose * getInverseCameraMatrix(k), nearPlane, farPlane, mu,
//        static_map_->get_volume_step(), static_map_->get_volume_step()*BLOCK_SIDE);
  raycast_mask_.reset();
//  std::cout<<"labelImg in raycasting type: "<<frame_masks_->labelImg.type()<<std::endl;
    raycastObjectList(objectlist_, vertex, normal, raycast_mask_.labelImg, objects_in_view_,
                      raycastPose, k, computationSize, nearPlane, farPlane,
                      mu, true);
    split_labels(raycast_mask_, objectlist_);

    if (in_debug_){
      mask_instance_output(raycast_mask_, objectlist_, frame,"raycast");
    }

    doRaycast = true;
//  }
  return doRaycast;
}


//use objectlist_ instead: integrate each volume separately
bool Kfusion::integration(float4 k, uint integration_rate, float mu,
                          uint frame) {

//  //on the first frame, only integrate backgrond
//  if (frame == 0){
//    cv::Mat bg_mask;
//    auto find_instance_mask = frame_masks_->instance_id_mask.find(0);
//    if ( find_instance_mask != frame_masks_->instance_id_mask.end()){
//      bg_mask = find_instance_mask->second;
//    }
//    else{
//      bg_mask = cv::Mat::ones(cv::Size(computationSize.y, computationSize.x), CV_8UC1);
//    }
//    static_map_->integrate_volume_kernel(floatDepth, inputRGB, bg_mask,
//                            computationSize, camera_pose, mu, k,
//                            frame);
//    return true;
//  }

//  bool doIntegrate = poses.empty() ? this->rgbdTracker_->checkPose(camera_pose,
//                                                                   T_w_r) : true;

  bool doIntegrate = true;
  if ((doIntegrate && ((frame % integration_rate) == 0)) || (frame <= 3)) {
    //single global volume integration
//    static_map_->integration_static_kernel(floatDepth, inputRGB,
//                                           computationSize, camera_pose, mu,
//                                           k, frame);
//    if (in_debug_) {
//      mask_instance_output(*frame_masks_, objectlist_, frame, "segment");
//    }
    //multiple objects tracking
    //integrate from the framemask direction
    //ENSURE: frame_masks share the SAME instance ID as objectlist
    for (auto object_mask = frame_masks_->pair_instance_seg_.begin();
        object_mask != frame_masks_->pair_instance_seg_.end(); ++object_mask){
      const int &instance_id = object_mask->first;
      const instance_seg& instance_mask = object_mask->second;
      if (instance_id == INVALID) continue;
//      const cv::Mat &instance_mask = object_mask->second.instance_mask_;
      const ObjectPointer& objectPtr = (objectlist_.at(instance_id));

      //create generalized instance mask
      cv::Mat gener_inst_mask;
      instance_mask.generalize_label(gener_inst_mask);

      //use motion residual to remove outliers in the generealized mask
      if (frame >1) {
        objectPtr->refine_mask_use_motion(gener_inst_mask,
                                          use_icp_tracking_,
                                          use_rgb_tracking_);
      }

      //      cv::Mat debug;
//      gener_inst_mask.convertTo(debug,CV_8U,255.0);
//      std::ostringstream name;
//      name << "/home/binbin/code/octree-lib-wip/apps/kfusion/cmake-build-debug/debug/"
//           <<"frame_"<< frame << "_object_id_" <<instance_id
//           <<"_gener_refine_mask.png";
//      cv::imwrite(name.str(), debug);


      //ensure instance_id is the same order in the objectlist
      //geometric fusion
      objectPtr->integrate_volume_kernel(floatDepth, inputRGB, gener_inst_mask,
                                         computationSize, camera_pose, mu, k,
                                         frame);

      //semantic fusion
      // no need to integrate background
      if (instance_mask.class_id_ == 0) continue;

        objectlist_[instance_id]->fuse_semantic_kernel(*frame_masks_,
                                                       instance_id);

    }


//  for (auto object = objectlist_.begin(); object != objectlist_.end();
//       ++object){
//    const int instance_id = (*object)->instance_label_;
//    cv::Mat labelMask;
//    auto find_instance_mask = frame_masks_->instance_id_mask.find(instance_id);
//      if ( find_instance_mask != frame_masks_->instance_id_mask.end()){
//        labelMask = find_instance_mask->second;
////        std::cout<<"labelMask in intergration type: "<<labelMask.type()<<std::endl;
//      }
//      else{
//        std::cout<<"in integration: mask missing: "<<instance_id<<std::endl;
//        labelMask = cv::Mat::ones(cv::Size(computationSize.y, computationSize.x), CV_8UC1);
////        continue;
//      }
//    (*object)->integrate_volume_kernel(floatDepth, inputRGB, labelMask,
//                                       computationSize, camera_pose, mu, k,
//                                       frame);
//  }

    doIntegrate = true;
  } else {
    doIntegrate = false;
  }

  return doIntegrate;

}

SegmentationResult Kfusion::volume2mask(float4 k, float mu){
  SegmentationResult renderedMask(computationSize.y, computationSize.x);
  volume2MaskKernel(objectlist_, renderedMask.labelImg, camera_pose, k,
                    computationSize, nearPlane, farPlane, mu);

  split_labels(renderedMask, objectlist_);
  renderedMask.set_render_label(true);
  return renderedMask;
}

bool Kfusion::MaskRCNN_next_frame(uint frame, std::string segFolder){
  if(segFolder == "") return true;
  if (mask_rcnn_result_.size()>frame) return true;
  return false;
}

bool Kfusion::readMaskRCNN(float4 k, uint frame, std::string segFolder){
//test if do-segment
  if(segFolder == ""){
    if(frame == segment_startFrame_){
      cv::Mat bg_mask = cv::Mat::ones(computationSize.y, computationSize.x,
                                      CV_8UC1)*255;
      instance_seg bg(0, bg_mask);
      frame_masks_->pair_instance_seg_.insert(std::make_pair(0, bg));
    }
    return false;
  }

  mask_rcnn = mask_rcnn_result_[frame];
  //oversegment current frame based on geometric edges
  geo_mask = segmenter_->compute_geom_edges(floatDepth, computationSize, k);

  if (in_debug_){
    std::cout<<"maskrcnn type: "<<mask_rcnn.pair_instance_seg_.begin()->second.instance_mask_.type() <<std::endl;
    mask_rcnn.output(frame, "maskrcnn");
  }

  //if there is no mask-rcnn recognision
  if (mask_rcnn.pair_instance_seg_.size() == 0) {
    //if initially
    if (frame == segment_startFrame_){
        std::cout << "mask-rcnn fails on the first frame, skip segmentation on "
                     "this frame" << std::endl;
        segment_startFrame_++;
    }
    return false;
  }
  //if there is mask-rcnn recognision
  else{

    if (geom_refine_human){
      //combine geometric components based on mask-rcnn
      std::cout << "Mask-RCNN has results, now combine it with geometric "
                   "components" << std::endl;
      if (do_edge_refinement){
        geo2mask_result.reset();
        segmenter_->mergeLabels(geo2mask_result, geo_mask, mask_rcnn,
                                segmenter_->geo2mask_threshold);
      }
      else{
        geo2mask_result = mask_rcnn;
      }

      if (in_debug_){
        geo2mask_result.output(frame, "geo2mask_result");
      }

      //not believe maskrcnn to remove human
      segmenter_->remove_human_and_small(human_out_, geo2mask_result,
                                         min_object_size_);
    }
    else{
      //believe maskrcnn to remove human
      segmenter_->remove_human_and_small(human_out_, mask_rcnn,
                                         min_object_size_);
    }




    if (in_debug_){
      human_out_.output(frame, "human_out");
    }

    return true;
  }
}


bool Kfusion::segment(float4 k, uint frame, std::string segFolder,
                      bool hasMaskRCNN){
  //test if do-segment
  if(segFolder == ""){
    /*if(frame == segment_startFrame_){
      cv::Mat bg_mask = cv::Mat::ones(computationSize.y, computationSize.x,
                                      CV_8UC1)*255;
      instance_seg bg(0, bg_mask);
      frame_masks_->pair_instance_seg_.insert(std::make_pair(0, bg));
    }*/
    return false;
  }

  if (use_GT_segment){
//    cv::Mat GT_mask = cv::imread(GT_mask_files_[frame]);
//        GT_mask = (GT_mask == 0);

    cv::Mat GT_label = cv::imread(GT_mask_files_[frame], CV_16SC1);

    cv::Mat GT_mask = (GT_label == 0);
    cv::cvtColor(GT_mask, GT_mask, CV_BGR2GRAY);
    //    GT_mask = (GT_mask == 7);
//    cv::Mat GT_label = cv::imread(GT_label_files_[frame]);
    if ((GT_mask.cols != static_cast<int>(computationSize.x)) ||
        (GT_mask.rows != static_cast<int>(computationSize.y)) ){
      cv::Size imgSize = cv::Size(computationSize.x, computationSize.y);
      cv::resize(GT_mask, GT_mask, imgSize);
    }
    SegmentationResult GT_segmentation(computationSize);
    instance_seg one_instance(58, GT_mask);
    GT_segmentation.pair_instance_seg_.insert(std::make_pair(1, one_instance));

    cv::Mat GT_mask1 = (GT_label == 2);
    cv::cvtColor(GT_mask1, GT_mask1, CV_BGR2GRAY);


//    GT_mask = (GT_mask == 7);
//    cv::Mat GT_label = cv::imread(GT_label_files_[frame]);
   if ((GT_mask1.cols != static_cast<int>(computationSize.x)) ||
        (GT_mask1.rows != static_cast<int>(computationSize.y)) ){
      cv::Size imgSize = cv::Size(computationSize.x, computationSize.y);
      cv::resize(GT_mask1, GT_mask1, imgSize);
    }
    SegmentationResult GT_segmentation1(computationSize);
   instance_seg one_instance1(57, GT_mask1);
    GT_segmentation.pair_instance_seg_.insert(std::make_pair(2, one_instance1));

    GT_segmentation.generate_bgLabels();
    frame_masks_ = std::make_shared<SegmentationResult>(GT_segmentation);

    if(frame == segment_startFrame_){
      generate_new_objects(GT_segmentation, k, frame);
    }

    return true;
  }

  if (hasMaskRCNN && (!geom_refine_human)){
    //combine geometric components based on mask-rcnn
    std::cout << "Mask-RCNN has results, now combine it with geometric "
                 "components" << std::endl;
    geo2mask_result = human_out_;
    if (do_edge_refinement)
      segmenter_->mergeLabels(geo2mask_result, geo_mask, human_out_,
                              segmenter_->geo2mask_threshold);

    if (in_debug_){
      geo2mask_result.output(frame, "geo2mask_result");
    }
  }
  else{
    geo2mask_result = human_out_;
  }



  //for the first frame, no extra object generated yet, no need to project
  // volumes to masks
  if(frame == segment_startFrame_){
    if (!hasMaskRCNN) {
      std::cout << "mask-rcnn fails on the first frame, skip segmentation on "
                   "this frame" << std::endl;
      segment_startFrame_++;
      return false;
    }

      //if there is mask-rcnn on the first frame, combine it with geometric
      // oversegmentation to create segmentation mask
    else{

        //combine geometric components based on mask-rcnn
//      SegmentationResult remingGeoMask(computationSize);
//        /*const SegmentationResult */geo2mask_result = segmenter_->mergeLabels(geo_mask,
//                                                                               mask_rcnn,
//                                                                               segmenter_->geo2mask_threshold);


//      std::cout << "Test: all prob->correspond to class" << std::endl;
//      mask_rcnn.print_class_all_prob();
//      geo2mask_result.print_class_all_prob();
//
//      if (in_debug_){
//        geo2mask_result.output(frame, "geo2mask_result");
//      }
//      SegmentationResult human_out(computationSize);
//
//      segmenter_->remove_human_and_small(human_out, geo2mask_result,
//                                         human_label, min_object_size_);
//
//      if (in_debug_){
//        human_out.output(frame, "human_out");
//      }
//
      if (!geom_refine_human) {
        geo2mask_result.exclude_human();
      }

      //on the first frame, only background model has already been generated
      SegmentationResult first_frame_mask(computationSize);
      const instance_seg& bg = geo2mask_result.pair_instance_seg_.at(0);
      first_frame_mask.pair_instance_seg_.insert(std::make_pair(0, bg));
      frame_masks_ = std::make_shared<SegmentationResult>(first_frame_mask);

      if (in_debug_){
        frame_masks_->output(frame, "frame_masks");
      }

      //if there is extra mask other than background generated on the first
      // frameif(frame_masks_->instance_id_mask.size()>1){
      generate_new_objects(geo2mask_result, k, frame);

      return true;
    }
  }



  if(frame>segment_startFrame_) {
    bool hasnewModel = false;

    //oversegment current frame based on geometric edges
//    /*SegmentationResult*/ geo_mask = segmenter_->compute_geom_edges(floatDepth,
//        computationSize, k);

    if (hasMaskRCNN) {
//      std::cout << "Mask-RCNN has results, now combine it with geometric "
//                   "components" << std::endl;

      //combine geometric components based on mask-rcnn
//      /*const SegmentationResult */geo2mask_result = segmenter_->mergeLabels(geo_mask,
//                                                                             mask_rcnn,
//                                                                             segmenter_->geo2mask_threshold);

//      if (in_debug_){
//        geo2mask_result.output(frame, "geo2mask_result");
//      }

      //render masks from object lists
//     /* *//*const SegmentationResult*//*
      if(raycast_mask_.pair_instance_seg_.size() <= 1){
        rendered_mask_ = volume2mask(k, config.mu);
        if (in_debug_) {
          rendered_mask_.output(frame,"volume2mask_old");
        }
      }


      SegmentationResult newModelMask(computationSize);
      SegmentationResult mask2model_result(computationSize);

      //compare the local segmentation to the projection from global objects
      hasnewModel = segmenter_->local2global(mask2model_result, newModelMask,
                                             geo2mask_result, rendered_mask_,
                                             min_object_size_,
                                             segmenter_->mask2model_threshold,
                                             segmenter_->new_model_threshold);

      //for the remaining that cannot match mask-rcnn (possibly due to wrong
      // recognition), try to match with the projected labels
      //std::cout << "start final merge" << std::endl;
      if (in_debug_) {
        mask2model_result.output(frame, "mask2model");
        newModelMask.output(frame, "newModelMask");
      }

      if (do_edge_refinement){
        SegmentationResult final_result = segmenter_->finalMerge(geo_mask,mask2model_result,
                                                                 segmenter_->geo2mask_threshold);
        final_result.exclude_human();
        frame_masks_ = std::make_shared<SegmentationResult>(final_result);
      }
      else{
        frame_masks_ = std::make_shared<SegmentationResult>(mask2model_result);
      }

      frame_masks_->combine_labels();


      if (in_debug_) {
      frame_masks_->output(frame, "frame_masks");
      }

      if(hasnewModel){
        generate_new_objects(newModelMask, k, frame);

      }
      return true;
    }
    else{
      if (in_debug_) {
        std::cout << "mask-rcnn fails on the frame " << frame
                  << ", now combine geometric components with projected labels"
                  << std::endl;
      }
      /*rendered_mask = volume2mask(k, mu);
      if (in_debug_) {
        mask_instance_output(rendered_mask, objectlist_, frame, "volume2mask");
      }*/

      SegmentationResult final_result = segmenter_->finalMerge(geo_mask,
                                                               rendered_mask_,
                                                               segmenter_->geo2mask_threshold);

      final_result.generate_bgLabels();
      frame_masks_ = std::make_shared<SegmentationResult>(final_result);

      //since there is no mask-rcnn, cannot judge if there is new object
      return true;
    }
  }

  if (in_debug_){
    for (auto object = objectlist_.begin(); object != objectlist_.end();
         ++object){
      int class_id = (*object)->class_id_;
      std::cout<<"after segment: class id: "<<class_id<<std::endl;
      printMatrix4("after segment: object pose ", (*object)->volume_pose_);
    }
  }

  return true;
}

void Kfusion::generate_new_objects(const SegmentationResult& masks,
                                   const float4 k, const uint frame){
  double start = tock();
  //has new label
  for (auto object_mask = masks.pair_instance_seg_.begin();
       object_mask!= masks.pair_instance_seg_.end(); ++object_mask) {

    const cv::Mat& mask =  object_mask->second.instance_mask_;
/*
    //threshold the size of mask to determine if a new object needs to be
        // generated
    int mask_size = cv::countNonZero(mask);
    if (mask_size< min_object_size_) {
      std::cout<<"object size "<<mask_size<<" smaller than threshold "
                                            <<min_object_size_<<std::endl;
      cv::bitwise_or(mask, masks.instance_id_mask[0], masks.instance_id_mask[0]);
      masks.instance_id_mask.erase(object_mask);
      continue;
    }
*/
    const int& instance_id = object_mask->first;
    const int& class_id = object_mask->second.class_id_;
    const All_Prob_Vect& class_all_prob = object_mask->second.all_prob_;

    if ((class_id == 0) || (class_id == 255)) continue;

//    ignore person:
//    if(class_id == 1) continue;
//    if(class_id == 57) continue;

    //determine the the volume size and pose
    float volume_size;
    Matrix4 T_w_o;
    int volume_resol;
    spawnNewObjectKernel(T_w_o, volume_size, volume_resol, mask, camera_pose, k);

    if (volume_size == 0) return;

    if (in_debug_){
      std::ostringstream name;
      name << "/home/binbin/code/octree-lib-wip/apps/kfusion/cmake-build-debug/debug/"
           <<"new_gen"<<"_frame_"<< frame << "_object_id_" <<instance_id
           <<"_class_id_" <<class_id<<".png";
      cv::imwrite(name.str(), mask);
    }

    //Matrix4 fake_pose = Identity();
    ObjectPointer new_object(new Object(config.voxel_block_size,
                                        make_float3(volume_size),
                                        make_uint3(volume_resol),
                                        T_w_o, camera_pose, class_id,
                                        class_all_prob,
                                        computationSize));

    //create generalized instance mask
    cv::Mat gener_inst_mask;
    object_mask->second.generalize_label(gener_inst_mask);

    new_object->integrate_volume_kernel(floatDepth, inputRGB, gener_inst_mask,
                                        computationSize, camera_pose, _mu, k, frame);

    std::cout<< "New object generated at frame "<<frame
             <<", with volume size: " <<new_object->get_volume_size().x
             <<", with volume resol: " <<new_object->get_volume_resol().x
             <<", with volume step: " <<new_object->get_volume_step()
             <<", class id: "<<class_id<< std::endl;
    printMatrix4(", with pose", T_w_o);
    add_object_into_list(new_object, this->objectlist_);
    objects_in_view_.insert(new_object->instance_label_);
//    if (instance_id>(int)objectlist_.size())
//      new_object->instance_label_ = instance_id;


  }

  init_time_ = tock() - start;
}

void Kfusion::spawnNewObjectKernel(Matrix4& T_w_o, float& volume_size,
                                   int& volume_resol,
                                   const cv::Mat& mask, const Matrix4& T_w_c,
                                   const float4&k){
  float3* c_vertex;
  c_vertex = (float3 *) calloc(sizeof(float3) * computationSize.x *
      computationSize.y, 1);
  depth2vertexKernel(c_vertex, floatDepth, computationSize,
                     getInverseCameraMatrix(k));

  float x_min = INFINITY;
  float y_min = INFINITY;
  float z_min = INFINITY;
  float x_max = -INFINITY;
  float y_max = -INFINITY;
  float z_max = -INFINITY;
  float x_avg = 0;
  float y_avg = 0;
  float z_avg = 0;
  int count = 0;

  std::cout<<computationSize.x<<" "<<computationSize.y<<std::endl;
  std::cout<<mask.size()<<std::endl;
//  std::cout<<"spawn mask type: "<<mask.type()<<std::endl;
  //TODO: openmp
  for (uint pixely = 0; pixely < computationSize.y; pixely++) {
    for (uint pixelx = 0; pixelx < computationSize.x; pixelx++) {
      if (mask.at<uchar>(pixely, pixelx) == 0) continue;
//      std::cout<<pixelx<<" "<<pixely<<std::endl;
      int id = pixelx + pixely * computationSize.x;
      float3 w_vertex = T_w_c * c_vertex[id];
      if (w_vertex.x > x_max) x_max = w_vertex.x;
      if (w_vertex.x < x_min) x_min = w_vertex.x;
      if (w_vertex.y > y_max) y_max = w_vertex.y;
      if (w_vertex.y < y_min) y_min = w_vertex.y;
      if (w_vertex.z > z_max) z_max = w_vertex.z;
      if (w_vertex.z < z_min) z_min = w_vertex.z;
      x_avg += w_vertex.x;
      y_avg += w_vertex.y;
      z_avg += w_vertex.z;
      count++;
    }
  }
  x_avg = x_avg/count;
  y_avg = y_avg/count;
  z_avg = z_avg/count;

//  x_avg = (x_max+x_min)/2.0f;
//  y_avg = (y_max+y_min)/2.0f;
//  z_avg = (z_max+z_min)/2.0f;

  std::cout<<"max/min x/y/z: "<<x_min<<" "<<x_max<<" "<<y_min<<" "<<y_max<<" "
           <<z_min<<" "<<z_max<<std::endl;
  float max_size = max(make_float3(x_max-x_min, y_max-y_min, (z_max-z_min)/2));
std::cout<<"average of vertex: "<<x_avg<<" "<<y_avg<<" "<<z_avg<<", with the max size " <<max_size <<std::endl;
  volume_size = fminf(2.5 * max_size, 5.0);


  //control one volume larger than 0.01cm
  if (volume_size < 0.64) volume_resol = 256;
  if ((0.64 < volume_size) && ( volume_size< 1.28)) volume_resol = 512;
  if ((1.28 < volume_size) && ( volume_size< 2.56)) volume_resol = 1024;
  if ((2.56 < volume_size) && ( volume_size< 5.12)) volume_resol = 2048;
//  if ((5.12 < volume_size) && ( volume_size< 10.24)) volume_resol = 512;


  //shift the T_w_o from center to left side corner
  T_w_o = Identity();
  T_w_o.data[0].w = x_avg - volume_size/2;
  T_w_o.data[1].w = y_avg - volume_size/2;
  T_w_o.data[2].w = z_avg - volume_size/2;
  free(c_vertex);
}

void Kfusion::add_object_into_list(ObjectPointer& f_objectpoint,
                                   ObjectList& f_objectlist){
  size_t current_object_nums = f_objectlist.size();
  f_objectpoint->instance_label_ = current_object_nums; //labels starting from 0
  f_objectlist.push_back(f_objectpoint);
};


void Kfusion::dumpVolume(std::string ) {

}

void Kfusion::printStats(){
  int occupiedVoxels = 0;
  for(unsigned int x = 0; x < static_map_->volume_._resol; x++){
    for(unsigned int y = 0; y < static_map_->volume_._resol; y++){
      for(unsigned int z = 0; z < static_map_->volume_._resol; z++){
        if( static_map_->volume_[make_uint3(x,y,z)].x < 1.f){
          occupiedVoxels++;
        }
      }
    }
  }
  std::cout << "The number of non-empty voxel is: " <<  occupiedVoxels << std::endl;
}

template <typename FieldType>
void raycastOrthogonal(Volume<FieldType> & volume, std::vector<float4> & points, const float3 origin, const float3 direction,
                       const float farPlane, const float step) {

  // first walk with largesteps until we found a hit
  auto select_depth =  [](const auto& val) { return val.x; };
  float t = 0;
  float stepsize = step;
  float f_t = volume.interp(origin + direction * t, select_depth);
  t += step;
  float f_tt = 1.f;

  for (; t < farPlane; t += stepsize) {
    f_tt = volume.interp(origin + direction * t, select_depth);
    if ( (std::signbit(f_tt) != std::signbit(f_t))) {     // got it, jump out of inner loop
      if(f_t == 1.0 || f_tt == 1.0){
        f_t = f_tt;
        continue;
      }
      t = t + stepsize * f_tt / (f_t - f_tt);
      points.push_back(make_float4(origin + direction*t, 1));
    }
    if (f_tt < std::abs(0.8f))               // coming closer, reduce stepsize
      stepsize = step;
    f_t = f_tt;
  }
}


void Kfusion::getPointCloudFromVolume(){

  std::vector<float4> points;

  float x = 0, y = 0, z = 0;

  int3 resolution = make_int3(static_map_->volume_._resol);
  float3 incr = make_float3(
      this->static_map_->volume_._size / resolution.x,
      this->static_map_->volume_._size / resolution.y,
      this->static_map_->volume_._size / resolution.z);

  // XY plane

  std::cout << "Raycasting from XY plane.. " << std::endl;
  for(y = 0; y < this->static_map_->volume_._size; y += incr.y ){
    for(x = 0; x < this->static_map_->volume_._size; x += incr.x){
      raycastOrthogonal(static_map_->volume_, points, make_float3(x, y, 0), make_float3(0,0,1),
                        this->static_map_->volume_._size, static_map_->get_volume_step());
    }
  }

  // ZY PLANE
  std::cout << "Raycasting from ZY plane.. " << std::endl;
  for(z = 0; z < this->static_map_->volume_._size; z += incr.z ){
    for(y = 0; y < this->static_map_->volume_._size; y += incr.y){
      raycastOrthogonal(static_map_->volume_, points, make_float3(0, y, z), make_float3(1,0,0),
                        this->static_map_->volume_._size, static_map_->get_volume_step());
    }
  }

  // ZX plane

  for(z = 0; z < this->static_map_->volume_._size; z += incr.z ){
    for(x = 0;  x < this->static_map_->volume_._size; x += incr.x){
      raycastOrthogonal(static_map_->volume_, points, make_float3(x, 0, z), make_float3(0,1,0),
                        this->static_map_->volume_._size, static_map_->get_volume_step());
    }
  }

  int num_points = points.size();
  std::cout << "Total number of ray-casted points : " << num_points << std::endl;

  if( !getenv("TRAJ")){
    std::cout << "Can't output the model point-cloud, unknown trajectory" << std::endl;
    return;
  }

  int trajectory = std::atoi(getenv("TRAJ"));
  std::stringstream filename;

  filename << "./pointcloud-vanilla-traj" << trajectory << "-" << static_map_->volume_._resol << ".ply";

  // Matrix4 flipped = toMatrix4( TooN::SE3<float>(TooN::makeVector(0,0,0,0,0,0)));

  // flipped.data[0].w =  (-1 * this->_initPose.x); 
  // flipped.data[1].w =  (-1 * this->_initPose.y); 
  // flipped.data[2].w =  (-1 * this->_initPose.z); 

  //std::cout << "Generating point-cloud.. " << std::endl;
  //for(std::vector<float4>::iterator it = points.begin(); it != points.end(); ++it){
  //        float4 vertex = flipped * (*it); 
  //    }
}
std::vector<uchar4> colors = random_color(91);

void Kfusion::renderVolume(uchar4 * out, uint2 outputSize, int frame,
                           int raycast_rendering_rate, float4 k, float
                           largestep, bool render_color) {
  if (frame % raycast_rendering_rate == 0){


//
//    renderVolumeKernel(static_map_->volume_, out, outputSize,
//                       *(this->viewPose) * getInverseCameraMatrix(k), nearPlane,
//                       farPlane * 2.0f, _mu, static_map_->get_volume_step(), largestep,
//                       get_translation(*(this->viewPose)), ambient,
//                       false, render_color_,
//                       vertex, normal);

//    Matrix4 poseTosee = camera_pose;
//  poseTosee.data[0].w -= 0.1f;
//    poseTosee.data[1].w -= 0.1f;
//    poseTosee.data[2].w -= 0.1f;
//        setViewPose(&poseTosee);
    renderVolume_many_Kernel(objectlist_, out, outputSize, *(this->viewPose), k,
                             nearPlane, farPlane * 2.0f, _mu, largestep, ambient,
                             (!compareMatrix4(*(this->viewPose), raycastPose)
                                 || (computationSize.x != outputSize.x) ||
                                 (computationSize.y != outputSize.y)),
                             render_color, vertex, normal,
                             raycast_mask_.labelImg, colors);

    if (render_output){
      std::string volume_file;
      if (render_color){
        volume_file = config.output_images+"color_volume";
      }
      else{
        volume_file = config.output_images+"label_volume";
      }
      opengl2opencv(out, outputSize, frame, volume_file);
    }
//    renderVolume_many_Kernel(objectlist_, out, outputSize, *(this->viewPose), k,
//                             nearPlane, farPlane * 2.0f, _mu, largestep, ambient,
//                             true,
//                             render_color_, vertex, normal,
//                             frame_masks_->labelImg);

  }


}

void Kfusion::renderTrack(uchar4 * out, uint2 outputSize) {

  if ((use_icp_tracking_) && (!use_rgb_tracking_)){
    if (raycast_mask_.pair_instance_seg_.size()>1){
      renderTrackKernel(out, objectlist_.at(1)->trackresult_, outputSize);
    }
    else{
      renderTrackKernel(out, objectlist_.at(0)->trackresult_, outputSize);
    }
  }
  if ((!use_icp_tracking_) && (use_rgb_tracking_)){
    if (raycast_mask_.pair_instance_seg_.size()>1){
      renderRGBTrackKernel(out, objectlist_.at(1)->trackresult_
          + computationSize.x * computationSize.y, outputSize);
    }
    else{
      renderRGBTrackKernel(out, objectlist_.at(0)->trackresult_
          + computationSize.x * computationSize.y, outputSize);
    }
  }
  if ((use_icp_tracking_) && (use_rgb_tracking_)){
    if (render_output){
//      renderTrackKernel(out, objectlist_.at(0)->trackresult_, outputSize);
//      cv::Mat icp_track = opengl2opencv(out, outputSize);
//      renderRGBTrackKernel(out, objectlist_.at(1)->trackresult_
//          + computationSize.x * computationSize.y, outputSize);
    }
    if (raycast_mask_.pair_instance_seg_.size()>1){
//      renderRGBTrackKernel(out, objectlist_.at(1)->trackresult_
//          + computationSize.x * computationSize.y, outputSize);
//      renderTrackKernel(out, objectlist_.at(1)->trackresult_, outputSize);

      render_RGBD_TrackKernel(out, objectlist_.at(1)->trackresult_,
                              objectlist_.at(1)->trackresult_ + computationSize
                                  .x * computationSize.y, outputSize);
    }
    else{
      render_RGBD_TrackKernel(out, objectlist_.at(0)->trackresult_,
                              objectlist_.at(0)->trackresult_ + computationSize
                                  .x * computationSize.y,
                              outputSize);
    }
  }
}


void Kfusion::renderTrack(uchar4 * out, uint2 outputSize, int type, int frame) {

  if (type == 0){
    if (raycast_mask_.pair_instance_seg_.size()>1){
      renderTrackKernel(out, objectlist_.at(0)->trackresult_, outputSize);
    }
    else{
      renderTrackKernel(out, objectlist_.at(0)->trackresult_, outputSize);
    }
    if (render_output){
      std::string motion_file;
      motion_file = config.output_images+"motion_icp";
      opengl2opencv(out, outputSize, frame, motion_file);
    }

  }
  if (type == 1){
    if (raycast_mask_.pair_instance_seg_.size()>1){
      renderRGBTrackKernel(out, objectlist_.at(1)->trackresult_
          + computationSize.x * computationSize.y, outputSize);
    }
    else{
      renderRGBTrackKernel(out, objectlist_.at(0)->trackresult_
          + computationSize.x * computationSize.y, outputSize);
    }
    if (render_output){
      std::string motion_file;
      motion_file = config.output_images+"motion_rgb";
      opengl2opencv(out, outputSize, frame, motion_file);
    }
  }
  if (type == 2){

    if (raycast_mask_.pair_instance_seg_.size()>1){
//      renderRGBTrackKernel(out, objectlist_.at(1)->trackresult_
//          + computationSize.x * computationSize.y, outputSize);
//      renderTrackKernel(out, objectlist_.at(1)->trackresult_, outputSize);

      render_RGBD_TrackKernel(out, objectlist_.at(0)->trackresult_,
                              objectlist_.at(0)->trackresult_ + computationSize
                                  .x * computationSize.y, outputSize);
    }
    else{
      render_RGBD_TrackKernel(out, objectlist_.at(0)->trackresult_,
                              objectlist_.at(0)->trackresult_ + computationSize
                                  .x * computationSize.y,
                              outputSize);
    }

    if (render_output){
      std::string motion_file;
      motion_file = config.output_images+"motion_joint_bg";
      opengl2opencv(out, outputSize, frame, motion_file);
    }

  }
}

void Kfusion::renderDepth(uchar4 * out, uint2 outputSize) {
  renderDepthKernel(out, floatDepth, outputSize, nearPlane, farPlane);
}

void Kfusion::renderDepth(uchar4 * out, uint2 outputSize, int frame) {
  renderDepthKernel(out, floatDepth, outputSize, nearPlane, farPlane);
  if (render_output){
    std::string depth_file;
    depth_file = config.output_images+"depth";
    opengl2opencv(out, outputSize, frame, depth_file);
  }
}

void Kfusion::renderIntensity(uchar4 * out, uint2 outputSize) {
  renderIntensityKernel(out, g_inputGrey, outputSize);
}


void Kfusion::renderClass(uchar4 *out,
                          uint2 outputSize,
                          SegmentationResult segmentationResult){
//  if (segmentationResult.class_id_mask.size()>0){
//    renderMaskKernel(out, g_inputGrey, outputSize, segmentationResult.labelImg, colors);
    renderClassMaskKernel(out, outputSize, segmentationResult, colors);
//  }
//  else{
//    std::cout<<"renderIntensity"<<std::endl;
//    renderIntensityKernel(out, g_inputGrey, outputSize);
//  }
}

void Kfusion::renderInstance(uchar4 * out, uint2 outputSize,
                             const SegmentationResult& segmentationResult) {
//  if (segmentationResult.class_id_mask.size()>0){
//    renderMaskKernel(out, g_inputGrey, outputSize, segmentationResult.labelImg, colors);
  renderInstanceMaskKernel(out, outputSize, segmentationResult, colors);
//  }
//  else{
//    std::cout<<"renderIntensity"<<std::endl;
//    renderIntensityKernel(out, g_inputGrey, outputSize);
//  }
}

void Kfusion::renderInstance(uchar4 * out, uint2 outputSize,
                             const SegmentationResult& segmentationResult,
                             int frame, std::string labelSource) {
//  if (segmentationResult.class_id_mask.size()>0){
//    renderMaskKernel(out, g_inputGrey, outputSize, segmentationResult.labelImg, colors);
  renderInstanceMaskKernel(out, outputSize, segmentationResult, colors);

  if (render_output){
    std::string mask_file;
    mask_file = config.output_images+labelSource;
    opengl2opencv(out, outputSize, frame, mask_file);
  }
}


void Kfusion::renderMaskWithImage(uchar4 * out, uint2 outputSize,
                                  const SegmentationResult&
                                  segmentationResult) {
  renderMaskWithImageKernel(out, outputSize, inputRGB, segmentationResult,
                            colors);
}


void Kfusion::renderMaskWithImage(uchar4 * out, uint2 outputSize,
                             const SegmentationResult& segmentationResult,
                                  int frame, std::string labelSource) {
  renderMaskWithImage(out,  outputSize, segmentationResult);

  if (render_output){
    std::string volume_file;
    volume_file = config.output_images+labelSource;
    opengl2opencv(out, outputSize, frame, volume_file);
  }
}

void Kfusion::renderMaskMotionWithImage(uchar4 * out, uint2 outputSize,
                                        const SegmentationResult& segmentationResult,
                                        int frame) {
  renderMaskMotionWithImageKernel(out, outputSize, inputRGB,
                                  segmentationResult, objectlist_, colors);

  if (render_output){
    std::string volume_file;
    volume_file = config.output_images+"inst_geo_motion";
    opengl2opencv(out, outputSize, frame, volume_file);
  }
}

void Kfusion::split_labels(SegmentationResult& segmentationResult,
                           const ObjectList& objectList){
  for (auto object = objectList.begin(); object != objectList.end(); ++object){
    const int &instance_id = (*object)->instance_label_;
    const int &class_id = (*object)->class_id_;
    const cv::Mat& mask = (segmentationResult.labelImg == instance_id);
    if (cv::countNonZero(mask) < 1) continue;
    instance_seg splitted(class_id, mask);
    segmentationResult.pair_instance_seg_.insert(std::make_pair(instance_id, splitted));
  }
}


void Kfusion::split_labels(SegmentationResult& segmentationResult,
                           std::set<int>& object_in_view,
                           const ObjectList& objectList){
  object_in_view.clear();
  for (auto object = objectList.begin(); object != objectList.end(); ++object){
    const int &instance_id = (*object)->instance_label_;
    const int &class_id = (*object)->class_id_;
    const cv::Mat& mask = (segmentationResult.labelImg == instance_id);
    if (cv::countNonZero(mask) < 1) continue;
    instance_seg splitted(class_id, mask);
    object_in_view.insert(instance_id);
    segmentationResult.pair_instance_seg_.insert(std::make_pair(instance_id, splitted));
  }
}

//void Kfusion::renderMask(uchar4 * out, uint2 outputSize, SegmentationResult segmentationResult) {
//  renderIntensityKernel(out, I_l[0], outputSize, segmentationResult.class_mask, segmentationResult.class_id, colors);
//}

//use objectlist_ instead: dump all volumes
void Kfusion::dump_mesh(const std::string filename){

  auto inside = [](const Volume<FieldType>::compute_type& val) {
    // meshing::status code;
    // if(val.y == 0.f) 
    //   code = meshing::status::UNKNOWN;
    // else 
    //   code = val.x < 0.f ? meshing::status::INSIDE : meshing::status::OUTSIDE;
    // return code;
    // std::cerr << val.x << " ";
    return val.x < 0.f;
  };

  auto select = [](const Volume<FieldType>::compute_type& val) {
    return val.x;
  };

  for(const auto& obj : objectlist_){
    std::vector<Triangle> mesh;
    algorithms::marching_cube(obj->volume_._map_index, select, inside,
                              mesh);
    const int obj_id = obj->instance_label_;
    const std::string obj_vol_name = filename+"_"+std::to_string(obj_id) +
        ".vtk";
    std::cout<<obj_vol_name<<std::endl;
    writeVtkMesh(obj_vol_name.c_str(), mesh);
  }

}


void Kfusion::save_poses(const std::string filename, const int frame){

  for(const auto& obj : objectlist_){
    const int obj_id = obj->instance_label_;
    Matrix4 obj_pose;
    std::string obj_pose_file;
    if (obj_id == 0) {
      obj_pose = camera_pose;
      obj_pose_file = filename+"_camera";
    }
    else{
      obj_pose = obj->volume_pose_;
      obj_pose_file = filename+"_"+std::to_string(obj_id);
    }
    Eigen::Quaternionf q = getQuaternion(obj_pose);
    std::ofstream ofs;
    if (obj->pose_saved_){
      ofs.open(obj_pose_file, std::ofstream::app);
    }
    else{
      ofs.open(obj_pose_file);
      obj->pose_saved_ = true;
    }

    if (ofs.is_open()){
//      save in the TUM pose: x-y-z-qx-qy-qz-qw
      ofs << frame << "\t" << camera_pose.data[0].w << "\t" << camera_pose.data[1].w
          << "\t" << camera_pose.data[2].w << "\t" << q.x() <<"\t"<<q.y()
          <<"\t"<<q.z() <<"\t"<<q.w() <<"\t"<< std::endl;
//      ofs.close();
    }
    else{
      std::cout << "Error opening file for object "<<obj_id<<std::endl;
    }
  }
}


void Kfusion::save_times(const std::string filename, const int frame,
                         double* timings){

  std::string time_file = filename+"_time";
  std::ofstream ofs;
  if(frame == 0){
    ofs.open(time_file);
  }
  else{
    ofs.open(time_file, std::ofstream::app);
  }

  size_t obj_num = objectlist_.size();
  size_t mov_obj_num = move_obj_set.size();
  size_t in_view_obj_num = objects_in_view_.size();
  double total = timings[8] - timings[0];
  double tracking = timings[4] - timings[3];
  double segmentation = timings[3] - timings[2] +timings[5] - timings[4];
  double integration = timings[6] - timings[5];
  double raycasting = timings[7] - timings[6];
  double computation = timings[7] - timings[2];

  if ((mov_obj_num == mov_obj_num_) && (in_view_obj_num == in_view_obj_num_)
      && (time_obj_num_ == obj_num)){
    //update
    total_ += total;
    tracking_ += tracking;
    segmentation_ += segmentation;
    integration_ += integration;
    raycasting_ += raycasting;
    computation_time_ += computation;
    same_obj_frame++;
  }
  else{
    //output
    if (ofs.is_open()){
//      save in the TUM pose: x-y-z-qx-qy-qz-qw
      ofs << time_obj_num_ << "\t"<< mov_obj_num_ << "\t"<< in_view_obj_num_ << "\t"
          << tracking_/same_obj_frame <<"\t"
          << segmentation_/same_obj_frame <<"\t"
          << init_time_ <<"\t"
          << integration_/same_obj_frame <<"\t"
          << raycasting_/same_obj_frame <<"\t"
          << computation_time_/same_obj_frame <<"\t"<< std::endl;
//      ofs.close();
    }
    else{
      std::cout << "Error opening file for time logging "<<std::endl;
    }

    //start new
    total_ = total;
    tracking_ = tracking;
    segmentation_ = segmentation;
    integration_ = integration;
    raycasting_ = raycasting;
    computation_time_ = computation;
    same_obj_frame = 1;
    init_time_ = 0;
    time_obj_num_ = obj_num;
    mov_obj_num_ = mov_obj_num;
    in_view_obj_num_ = in_view_obj_num;
  }



}

void synchroniseDevices() {
  // Nothing to do in the C++ implementation
}

