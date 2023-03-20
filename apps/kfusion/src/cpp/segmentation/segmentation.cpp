 /*
 * SPDX-FileCopyrightText: 2017-2019 Smart Robotics Lab, Imperial College London
 * SPDX-FileCopyrightText: 2017-2019 Binbin Xu
 * SPDX-License-Identifier: BSD-3-Clause
 */


#include "segmentation.h"

void instance_seg::generalize_label(cv::Mat& gener_inst_mask) const{

  cv::Size maskSize = instance_mask_.size();
  cv::Mat element = cv::getStructuringElement(cv::MORPH_ELLIPSE,
                                              cv::Size(bg_dilute_size_,
                                                       bg_dilute_size_));

  cv::Mat mask_with_bg, out_bg;
  cv::dilate(instance_mask_, mask_with_bg, element);

  if (this->hasModelMask_ == true){
    cv::bitwise_or(mask_with_bg, global_mask_, mask_with_bg);
  }
  cv::bitwise_not(mask_with_bg, out_bg);
  cv::Mat bg_layer = mask_with_bg - instance_mask_;

  //if the mask is generated from mask-rcnn
  if (!rendered_){
    gener_inst_mask = cv::Mat::ones(maskSize, CV_32FC1);
  }
  else{//if mask is rendered from volumes
    int mask_size = cv::countNonZero(instance_mask_);
    float mask_threshold = render_mask_size_ratio_ * instance_mask_.cols *
        instance_mask_.rows;
    //if the rendered mask is small
    if(mask_size < mask_threshold){
      gener_inst_mask = cv::Mat::ones(maskSize, CV_32FC1) * small_render_fg_;
    }
      //if the rendered mask is large
    else{
      gener_inst_mask = cv::Mat::ones(maskSize, CV_32FC1) * large_render_fg_;
    }
  }

  cv::Mat label_out = cv::Mat::ones(maskSize, CV_32FC1) * -1;
  label_out.copyTo(gener_inst_mask, out_bg);
  cv::Mat bg_layer_mask = cv::Mat::zeros(maskSize, CV_32FC1);
  bg_layer_mask.copyTo(gener_inst_mask, bg_layer);
}

void instance_seg::merge(const instance_seg& new_instance){
  //ensure the merged two share the same class id
  if(this->class_id_ != new_instance.class_id_) return;

  cv::bitwise_or(new_instance.instance_mask_, this->instance_mask_,
                 this->instance_mask_);
  this->all_prob_ = (new_instance.all_prob_ * new_instance.recog_time_ +
      this->all_prob_ * this->recog_time_) /
      (this->recog_time_ + new_instance.recog_time_);

  this->recog_time_ += new_instance.recog_time_;

}

//instance_seg SegmentationResult::find_segment_from_instID(const int instance_id,
//                                                    bool info) const{
//
//  auto find_class_id = this->pair_instance_seg_.find(instance_id);
//  if (find_class_id == this->pair_instance_seg_.end()){
//    if(info){
//      std::cerr<<"segmentation missing for instance id: "
//                 ""<<instance_id<<std::endl;
//      exit(0);
//    }
//  }
//  else{
//    instance_seg segmentation = find_class_id->second;
//    return segmentation;
//  }
//}

void SegmentationResult::print_class_all_prob() const{
  for (auto prob_it = pair_instance_seg_.begin(); prob_it !=
      pair_instance_seg_.end(); prob_it++){
//    std::cout<<"------------------------------------"<<std::endl;
//    std::cout<<"instance id: "<<prob_it->first<<" :" <<prob_it->second
//        .all_prob_<<std::endl;

    All_Prob_Vect::Index max_id;
    float max_prob = prob_it->second.all_prob_.maxCoeff(&max_id);
//    std::cout<<"its class id is: "<<(max_id+1)<<" with prob:"<<max_prob<<std::endl;
//    std::cout<<"directly input class id is: " <<prob_it->second.class_id_ <<std::endl;
  }
}

////return the all probablity foir that instance id
////if the instance id is not detected on this frame, return false
//bool SegmentationResult::find_classProb_from_instID(All_Prob_Vect& all_prob,
//                                                    const int &instance_id,
//                                                    bool info) const{
//  bool class_recoged=false;
//
//  auto find_class_all_prob = this->instance_id_all_prob_.find(instance_id);
////  std::cout<<find_class_all_prob->second<<std::endl;
////  std::cout<<find_class_all_prob->second.size()<<" "
////                                                 ""<<find_class_all_prob->second.rows()<<" "<<find_class_all_prob->second.cols()<<std::endl;
//
////  std::cout<<class_all_prob.size()<<" "<<class_all_prob.rows()<<" "<<class_all_prob.cols()<<std::endl;
//
//  if (find_class_all_prob == this->instance_id_all_prob_.end()){
//    if(info){
//      std::cout<<"class prob missing for instance id: "<<instance_id<<std::endl;
//    }
//  }
//  else{
//    all_prob = find_class_all_prob->second;
//    class_recoged = true;
//  }
//  return class_recoged;
//}

//void SegmentationResult::merge_mask_rcnn(const float overlap_ratio){
//  for (auto object = this->instance_class_id.begin();
//       object != this->instance_class_id.end(); ++object) {
//    //skip the beginning
//    if (object == this->instance_class_id.begin()) continue;
//
//  }
//}

void SegmentationResult::set_render_label(const bool rendered){
  for (auto instance_it = pair_instance_seg_.begin(); instance_it !=
      pair_instance_seg_.end(); instance_it++){
    instance_it->second.rendered_ = rendered;
    instance_it->second.recog_time_ = 0;
  }
}

void SegmentationResult::output(const uint frame, const std::string& str )
const{
  for (auto instance_it = pair_instance_seg_.begin(); instance_it !=
      pair_instance_seg_.end(); instance_it++){
    const int& index = instance_it->first;
    const cv::Mat &labelMask = instance_it->second.instance_mask_;
    const int& class_id = instance_it->second.class_id_;
    if ( class_id > 0) {
      std::ostringstream name;
      name << "/home/binbin/code/octree-lib-wip/apps/kfusion/cmake-build-debug/debug/"
           <<str<<"_frame_"<< frame << "_object_id_" <<index
           <<"_class_id_" <<class_id<<".png";
      cv::imwrite(name.str(), labelMask);
    } else {
      std::ostringstream name;
      name << "/home/binbin/code/octree-lib-wip/apps/kfusion/cmake-build-debug/debug/"
           <<str<<"_frame_"<< frame << "_object_id_" <<index<<".png";
      cv::imwrite(name.str(), labelMask);
    }
  }
}

void SegmentationResult::exclude_human() {
  //ensure the masks are excluding the human area
  if (this->pair_instance_seg_.find(INVALID) !=
      this->pair_instance_seg_.end()){
    const cv::Mat human_mask = this->pair_instance_seg_.at(INVALID)
        .instance_mask_;
    cv::Mat not_human;
    cv::bitwise_not(human_mask, not_human);

    for(auto output_instance = this->pair_instance_seg_.begin();
        output_instance != this->pair_instance_seg_.end();
        output_instance++){
      const int output_instance_id = output_instance->first;

      //if human, skip
      if (output_instance_id == INVALID) continue;
      else{
        cv::Mat& output_instance_mask = output_instance->second.instance_mask_;
        cv::bitwise_and(output_instance_mask, not_human, output_instance_mask);
      }
    }
  }
}

std::vector<std::string> Segmentation::readFiles(const std::string file_dir){
  char * file_folder = new char [file_dir.length()+1];
  strcpy (file_folder, file_dir.c_str());

  DIR *dir;
  struct dirent *ent;
  if ((dir = opendir (file_folder)) != NULL) {
    std::cout<<"reading folder: "<<file_dir<<std::endl;
    std::vector<std::string> local_files;
    /* print all the files and directories within directory */
    while ((ent = readdir (dir)) != NULL) {
      //skip folder of '.' and '..'
      if(ent->d_name[0] == '.')
        continue;
      else if(ent->d_type == 8)    ///file
        //printf("d_name:%s/%s\n",basePath,ptr->d_name);
        local_files.push_back(ent->d_name);
      else if(ent->d_type == 10)    ///link file
        //printf("d_name:%s/%s\n",basePath,ptr->d_name);
        continue;
      else if(ent->d_type == 4)    ///dir
      {
        local_files.push_back(ent->d_name);
      }
//        printf ("%s\n", ent->d_name);
    }
    sort(local_files.begin(), local_files.end());
    std::vector<std::string> abs_files;
    for (std::vector<std::string>::const_iterator i = local_files.begin(); i != local_files.end(); ++i){
      abs_files.push_back(file_dir + *i);
    }

    //print to see what is read
//    for (std::vector<std::string>::const_iterator i = abs_files.begin(); i != abs_files.end(); ++i){
//      std::cout<<*i<<std::endl;
//    }
    delete[] file_folder;
    closedir (dir);
    return abs_files;
  }
  else{
    std::cerr<<"\033[31mERROR: Mask RCNN directory does not exist or is not given.\033[0m" <<std::endl;
    std::cout<<"The passed Mask RCNN directory path is:\n" << file_folder << std::endl;
    exit (EXIT_FAILURE);
  }
}

std::vector<int> Segmentation::load_class_ids(const std::string class_path){
//  std::string class0_path = class_files[91];
  cnpy::NpyArray class_id_npy = cnpy::npy_load(class_path);

  //make sure the loaded data matches the saved data
  if(class_id_npy.word_size != sizeof(int64)){
    std::cerr<<class_path<<"=>cnpy class-idx reading format is wrong"<<std::endl;
    std::cerr<<"cnpy format: "<<class_id_npy.word_size<<std::endl;
    exit (EXIT_FAILURE);
  };
  int64 * class_id_array = class_id_npy.data<int64>();

  std::vector<int> class_id;
//  std::cout<<class_path;
  for(int i = 0; i < class_id_npy.shape[0];i++){
    class_id.push_back(static_cast<int>(class_id_array[i]));
//    std::cout<<" "<<class_id[i];
  }
//  std::cout<<" "<<class_id.size()<<std::endl;
  return class_id;
}

Eigen::MatrixXf Segmentation::readNPY(const std::string& filename)
{
  cnpy::NpyArray arr = cnpy::npy_load(filename);
  float* loaded_data = arr.data<float>();
  assert(arr.word_size == sizeof(float));
  assert(arr.shape.size() == 2);
  uint rows = arr.shape[0];
  uint cols = arr.shape[1];
  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> mat(rows, cols);
  uint counter = 0;
  for (uint i = 0; i < rows; i++)
  {
    for (uint j = 0; j < cols; j++)
    {
      mat(i, j) = loaded_data[counter];
      counter++;
    }
  }
  return mat;
}

std::vector<All_Prob_Vect> Segmentation::load_class_prob(const std::string
                                                  prob_path, uint& class_num){
//  std::string class0_path = class_files[91];
  Eigen::MatrixXf prob_npy = readNPY(prob_path);
  //cnpy::NpyArray class_prob_npy = cnpy::npy_load(prob_path);
  std::vector<All_Prob_Vect> class_prob;

  int instance_num = prob_npy.rows();
  if (instance_num==0) return  class_prob;

  /*
  //make sure the loaded data matches the saved data
  if(class_prob_npy.word_size != sizeof(float)){
    std::cerr<<prob_path<<"=>cnpy class-idx reading format is wrong"<<std::endl;
    std::cerr<<"cnpy format: "<<class_prob_npy.word_size<<std::endl;
    exit (EXIT_FAILURE);
  };
   */

  //float * class_prob_array = class_prob_npy.data<float>();
  class_num = prob_npy.cols();
//  if (class_num != 80){
//    std::cerr<<"class number is wrong"<<std::endl;
//    exit (EXIT_FAILURE);
//  }

//  std::cout<<prob_npy.size()<<" "<<prob_npy.cols()<<" "<<prob_npy.rows()
//           <<std::endl<<prob_npy<<std::endl;

  class_prob.reserve(instance_num);
//  std::cout<<class_path;
  for(int i = 0; i < instance_num; i++){
    All_Prob_Vect class_all_prob = prob_npy.row(i);
//    std::cout<<class_all_prob.size()<<" "<<class_all_prob.rows()<<" "<<class_all_prob.cols()
//             <<std::endl<<class_all_prob<<std::endl;
    class_prob.push_back(class_all_prob);
    //free(class_all_prob);
//    std::cout<<" "<<class_id[i];
  }
//  std::cout<<" "<<class_id.size()<<std::endl;
  return class_prob;
}

std::vector<cv::Mat> Segmentation::load_mask(const std::string mask_path, const int width, const int height){

  //return
  std::vector<cv::Mat> idx_pixel_mask;

//  for debug frame information
//  std::cout<<mask_path<<std::endl;

  //load mask
  cnpy::NpyArray mask_npy = cnpy::npy_load(mask_path);

  if (mask_npy.shape[0]==0) return idx_pixel_mask;

  if(mask_npy.word_size != sizeof(bool)){
    //    no prediction
    if(mask_npy.shape[2]==0){
      cv::Mat no_idx = cv::Mat::zeros(cv::Size(height, width), CV_8UC1);
      idx_pixel_mask.push_back(no_idx);
      return idx_pixel_mask;
    }
    else{
      std::cerr<<mask_path<<"=>cnpy mask reading format is wrong"<<std::endl;
      std::cerr<<"cnpy format: "<<mask_npy.word_size<<std::endl;
      std::cerr<<"Given mask: ("<<mask_npy.shape[0]<<", "<<mask_npy.shape[1]<<", "<<mask_npy.shape[2]<<")"<<std::endl;
      exit (EXIT_FAILURE);
    }
  };

  if (mask_npy.shape.size() != 3){
    std::cerr<<"mask shape is wrong, not in 3D!"<<std::endl;
    exit (EXIT_FAILURE);
  }

  //resize now:
//  if ((mask_npy.shape[0]!= width) || (mask_npy.shape[1] != height)){
//    std::cerr<<"Given mask SIZE is not matching with the SIZE of frame!"<<std::endl;
//    std::cerr<<"Given mask: ("<<mask_npy.shape[0]<<", "<<mask_npy.shape[1]<<")"<<std::endl;
//    std::cerr<<"Input size: ("<<width<<", "<<height<<")"<<std::endl;
//    exit (EXIT_FAILURE);
//  }

  idx_pixel_mask.reserve(mask_npy.shape[0]);


  for (int id = 0; id< mask_npy.shape[0]; id++){
    bool * mask_pixel = (bool*)malloc(sizeof(bool) * mask_npy.shape[1] *
        mask_npy.shape[2]);
    load_mask_kernel(mask_pixel, mask_npy, id);
    cv::Mat mask_map(cv::Size(mask_npy.shape[2], mask_npy.shape[1]), CV_8UC1);
    std::memcpy(mask_map.data, mask_pixel,
                mask_npy.shape[1] * mask_npy.shape[2] *sizeof(bool));
    mask_map = mask_map * 255;
    if ((mask_npy.shape[2]!= width) || (mask_npy.shape[1] != height)){
      //if size is not matching: resize
      cv::resize(mask_map, mask_map, cv::Size(height, width));
    }
      idx_pixel_mask.push_back(mask_map);
    free(mask_pixel);
  }


  return idx_pixel_mask;
}

void Segmentation::load_mask_kernel(bool * mask_pixel, cnpy::NpyArray mask_npy, int instance_id){
  bool * mask_array = mask_npy.data<bool>();
    unsigned int y;
#pragma omp parallel for \
        shared(mask_pixel), private(y)
    for (y = 0; y < mask_npy.shape[1]; y++) {
      for (unsigned int x = 0; x < mask_npy.shape[2]; x++) {
        unsigned int mask_pos = y * mask_npy.shape[2] + x;
//        unsigned int img_pos = x * height + y;
        unsigned int npy_pos = mask_npy.shape[1]*mask_npy.shape[2]*instance_id + mask_pos;
        mask_pixel[mask_pos] = mask_array[npy_pos];
      }
    }
  }

void SegmentationResult::combine_labels(){
  if (this->pair_instance_seg_.size()>0){
    cv::Size imgSize = this->pair_instance_seg_.begin()->second
        .instance_mask_.size();
    cv::Mat combined_labels = cv::Mat::zeros(imgSize, CV_32SC1);;

    for (auto it = this->pair_instance_seg_.begin();
        it != this->pair_instance_seg_.end(); ++it) {
      cv::Mat labelMat = cv::Mat::ones(imgSize, CV_32SC1) * it->first;
      labelMat.copyTo(combined_labels, it->second.instance_mask_);
    }
    this->labelImg = combined_labels;
  }
}

void SegmentationResult::generate_bgLabels(){
  if(this->pair_instance_seg_.find(0) != this->pair_instance_seg_.end()){
    this->pair_instance_seg_.erase(0);
  }

  cv::Mat bg_mask= cv::Mat::ones(width_, height_, CV_8UC1)*255;
  for (auto object = this->pair_instance_seg_.begin();
      object != this->pair_instance_seg_.end() ; ++object) {
    cv::Mat object_out;
    cv::bitwise_not(object->second.instance_mask_, object_out);
    cv::bitwise_and(bg_mask, object_out, bg_mask);
  }

  instance_seg bg_seg(0, bg_mask);
  this->pair_instance_seg_.insert(std::make_pair(0, bg_seg));
  this->combine_labels();
}

SegmentationResult Segmentation::load_mask_rcnn(const std::string class_path,
                                                const std::string mask_path,
                                                const std::string prob_path,
                                                const int width, const int height){
  SegmentationResult mask_rcnn_segmentation(width, height);
  std::vector<int> ids = load_class_ids(class_path);

  //if no recognision
  if (ids.size() == 0) {
    mask_rcnn_segmentation.generate_bgLabels();
    return mask_rcnn_segmentation;
  }

  std::vector<All_Prob_Vect> probs = load_class_prob(prob_path,
      mask_rcnn_segmentation.label_number_);
  std::vector<cv::Mat> masks = load_mask(mask_path, width, height);
  for (int it = 0; it<ids.size(); ++it){
    //0 instance id matches the 0 class id since ids is in order
    //filter dinner table...
    if (filter_class_id_.find(ids[it]) != filter_class_id_.end()){
      continue;
    }
    instance_seg one_instance(ids[it], masks[it], probs[it]);
    mask_rcnn_segmentation.pair_instance_seg_.insert(std::make_pair(it+1, one_instance));
  }
  mask_rcnn_segmentation.generate_bgLabels();
  return mask_rcnn_segmentation;
}

SegmentationResult Segmentation::compute_geom_edges(const float * depth, const uint2 inSize, const float4 k){

  SegmentationResult geometric_segmentation(inSize.y, inSize.x);
  bool* notEdge = (bool*) calloc(sizeof(bool) * inSize.x * inSize.y, 1);

  Matrix4 invK = getInverseCameraMatrix(k);
  float3 * vertex = (float3*) calloc(sizeof(float3) * inSize.x * inSize.y, 1);
  float3 * normal = (float3*) calloc(sizeof(float3) * inSize.x * inSize.y, 1);


  depth2vertexKernel(vertex, depth, inSize, invK);

  if(k.y < 0)
    vertex2normalKernel<FieldType, true>(normal, vertex, inSize);
  else
    vertex2normalKernel<FieldType, false>(normal, vertex, inSize);

  geometric_edge_kernel(notEdge, vertex, normal, inSize, geometric_lambda, geom_threshold);
  cv::Mat noedge_map(inSize.y, inSize.x, CV_8UC1);
  std::memcpy(noedge_map.data, notEdge, inSize.x*inSize.y*sizeof(bool));

  //add rectangular boundary
//  cv::rectangle(edge_map,cv::Point(0,0), cv::Point(edge_map.cols, edge_map.rows), true);
//
//////  fill in boundary
//  std::vector<std::vector<cv::Point>> edge_contours;
//  cv::findContours(edge_map.clone(), edge_contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
//
//  cv::Mat edge_filled(edge_map.size(), CV_8U);
//  cv::drawContours(edge_filled, edge_contours, -1, cv::Scalar(255), CV_FILLED);

  cv::Mat geomtric_label;
  int connectivity=4;
  cv::Mat label_stats;
  cv::Mat label_centroids;
  int nLabels = cv::connectedComponentsWithStats(noedge_map, geomtric_label, label_stats, label_centroids,
                                                 connectivity);
//  component 0 is background
//      geometric_segmentation.class_id.push_back(0);
//      bool* maskImage = (bool*) calloc(sizeof(bool) * inSize.x * inSize.y, 1);
//      label2mask(maskImage, geomtric_label, 0, inSize);
//      geometric_segmentation.class_mask.push_back(maskImage);

  geometric_segmentation.labelImg = geomtric_label.clone();

  for (int label_id = 0; label_id < nLabels; ++label_id) {
    if (label_stats.at<int>(label_id, cv::CC_STAT_AREA)>geomtric_component_size){
      cv::Mat maskImage = (geomtric_label == label_id);

      //dirty: dilate mask a little bit
      cv::Mat element = cv::getStructuringElement( cv::MORPH_ELLIPSE,cv::Size
          (geom_dilute_size, geom_dilute_size));
      cv::dilate(maskImage, maskImage, element );
//      std::cout<<"geo_mask id: "<<label_id
//               <<" size: "<<maskImage.size()<<std::endl;
      instance_seg geo_seg(maskImage);
      geometric_segmentation.pair_instance_seg_.insert(std::make_pair(label_id, geo_seg));
    }
  }

  free(vertex);
  free(normal);
  free(notEdge);
  return geometric_segmentation;
}


float Segmentation::calc_distance(const float3* inVertex, const float3* inNormal, const int id, const int id_i){
  if (inNormal[id] == INVALID || inNormal[id_i] == INVALID) {
    return -1;
  }

  return dot(inNormal[id], inVertex[id_i] - inVertex[id]);
}

float Segmentation::calc_concavity(const float3* inNormal, const int id, const int id_i){
  if (inNormal[id] == INVALID || inNormal[id_i] == INVALID) {
    return -1;
  }

  return 1 - dot(inNormal[id], inNormal[id_i]);
}

void Segmentation::geometric_edge_kernel(bool * notEdge, const float3* inVertex, const float3* inNormal,
                            const uint2 inSize, const float lambda, const float threshold){
  uint2 pixel = make_uint2(0, 0);
  unsigned int pixely, pixelx;
#pragma omp parallel for \
	    shared(notEdge), private(pixel,pixelx,pixely)
  for (pixely = 0; pixely < inSize.y; pixely++) {
    for (pixelx = 0; pixelx < inSize.x; pixelx++) {
      pixel.x = pixelx;
      pixel.y = pixely;
      uint id = pixelx + pixely * inSize.x;

//      add rectangular boundary
      if ((pixelx == 0) && (0< pixely < inSize.y-2) || (pixelx == inSize.y-1) && (0< pixely < inSize.y-2) ||
        (pixely == 0) && (0< pixelx < inSize.x-2) || (pixely == inSize.y-1) && (0< pixelx < inSize.x-2)){
        notEdge[id] = false;
        continue;
      }

      std::vector<float> distance;
      std::vector<float> concavity;

      for(int i = pixelx -1; i <= pixelx + 1; ++i){
        for (int j = pixely -1; j <= pixely + 1; ++j) {
          if ((i != pixelx) && (j != pixely) && (0<=i) && (i<inSize.x) && (0<=j) && (j<inSize.y))
          {
            const int id_i = i + j * inSize.x;
            const float distance_i = calc_distance(inVertex, inNormal, id, id_i);
            distance.push_back(fabsf(distance_i));

            float concavity_i;
            if(distance_i<0) {concavity_i =0;}
            else{concavity_i = calc_concavity(inNormal, id, id_i);}
            concavity.push_back(concavity_i);
          }
        }
      }

      float max_concave, max_distance, edgeness;
      if (distance.size()>0) max_concave = *std::max_element(concavity.begin(), concavity.end());
      if (distance.size()>0) max_distance = *std::max_element(distance.begin(), distance.end());
      if ((distance.size()>0) && (distance.size()>0)) edgeness = max_distance + lambda * max_concave;

      if (edgeness > threshold){notEdge[id] = false;}
      else{notEdge[id] = true;}
    }
  }
}

void Segmentation::mergeLabels(SegmentationResult& mergedSeg,
                               const SegmentationResult& srcSeg,
                               const SegmentationResult& dstSeg,
                               const float threshold){
//  SegmentationResult mergedSeg(srcSeg.width_, srcSeg.height_);
//  SegmentationResult mergedSeg = dstSeg;
  for (auto src_mask = srcSeg.pair_instance_seg_.begin();
      src_mask != srcSeg.pair_instance_seg_.end(); ++src_mask) {
    float max_overlap = 0;
    const cv::Mat& src_mask_mat = src_mask->second.instance_mask_;
    const std::pair<const int, instance_seg>* overlap_dst_mask_ptr;

    for (auto dst_mask = dstSeg.pair_instance_seg_.begin();
        dst_mask != dstSeg.pair_instance_seg_.end() ; ++dst_mask) {
      const int &dst_label = dst_mask->first;
      const cv::Mat& dst_mask_mat = dst_mask->second.instance_mask_;
      if (dst_label == 0) continue;
      if (!refine_human_boundary_){
        if (dst_label == INVALID) continue;
      }
//      std::cout<<"src_mask_mat: "<<src_mask_mat.size()<<std::endl;
//      std::cout<<"dst_mask_mat: "<<dst_mask_mat.size()<<std::endl;

      cv::Mat overlap_mask;
      cv::bitwise_and(src_mask_mat, dst_mask_mat, overlap_mask);
      const float ratio = static_cast<float>(cv::countNonZero(overlap_mask)) /
        static_cast<float>(cv::countNonZero(src_mask_mat));
      if (ratio>max_overlap) {
        max_overlap = ratio;
        overlap_dst_mask_ptr = &(*dst_mask);
      }
      else{
        continue;
      }

    }
    if (max_overlap>threshold){
      //look for that dst class
      const int &overlap_inst_id = overlap_dst_mask_ptr->first;
      auto find_merged_id = mergedSeg.pair_instance_seg_.find(overlap_inst_id);
      // not that class has not been merged before, insert a new source component
      if ( find_merged_id == mergedSeg.pair_instance_seg_.end()) {
        mergedSeg.pair_instance_seg_.insert(*overlap_dst_mask_ptr);
        mergedSeg.pair_instance_seg_.at(overlap_inst_id).instance_mask_ = src_mask_mat;
      }
        // the class already a mask component, merged them together
      else {
        //update the mask
        cv::bitwise_or(src_mask_mat, find_merged_id->second.instance_mask_,
                       find_merged_id->second.instance_mask_);
      }
    }
//    else{
//      remingSrcSeg.class_id_mask.insert(*src_mask);
//    }
  }

  //generate combined label masks
  mergedSeg.generate_bgLabels();
}

SegmentationResult Segmentation::finalMerge(const SegmentationResult& srcSeg,
                                             const SegmentationResult& dstSeg,
                                             const float threshold){
  SegmentationResult mergedSeg = dstSeg;

  for (auto src_instance = srcSeg.pair_instance_seg_.begin();
      src_instance != srcSeg.pair_instance_seg_.end(); ++src_instance) {
    float max_overlap = 0;
    const cv::Mat& src_mask_mat = src_instance->second.instance_mask_;
    const std::pair<const int, instance_seg>* overlap_dst_mask_ptr;
    for (auto dst_instance = dstSeg.pair_instance_seg_.begin();
        dst_instance != dstSeg.pair_instance_seg_.end() ; ++dst_instance) {
      const cv::Mat& dst_mask_mat = dst_instance->second.instance_mask_;
      const int &dst_label = dst_instance->first;
      if (dst_label == 0) continue;
      if (!refine_human_boundary_){
        if (dst_label == INVALID) continue;
      }
      //      std::cout<<"src_mask_mat: "<<src_mask_mat.size()<<std::endl;
//      std::cout<<"dst_mask_mat: "<<dst_mask_mat.size()<<std::endl;

      cv::Mat overlap_mask;
      cv::bitwise_and(src_mask_mat, dst_mask_mat, overlap_mask);
      const float ratio = static_cast<float>(cv::countNonZero(overlap_mask)) /
          static_cast<float>(cv::countNonZero(src_mask_mat));
      if (ratio>max_overlap) {
        max_overlap = ratio;
        overlap_dst_mask_ptr = &(*dst_instance);
      }
      else{
        continue;
      }
    }
    if (max_overlap>threshold){
      //look for that dst class
      const int &overlap_inst_id = overlap_dst_mask_ptr->first;
      auto merged_id_find = mergedSeg.pair_instance_seg_.find(overlap_inst_id);

      // if that class has not been merged before, insert a new source component
      if ( merged_id_find == mergedSeg.pair_instance_seg_.end()) {
//        mergedSeg.instance_id_mask.insert(std::make_pair(overlap_inst_id, src_mask_mat));
//        auto merged_instance_class_id = dstSeg.instance_class_id.find(overlap_inst_id);
//        auto merged_instance_class_prob = dstSeg.instance_id_all_prob_.find(overlap_inst_id);
//        mergedSeg.instance_class_id.insert(*merged_instance_class_id);
//        mergedSeg.instance_id_all_prob_.insert(*merged_instance_class_prob);
      }
        // the class already a mask component, merged them together
      else {
        //update the mask
        cv::bitwise_or(src_mask_mat, merged_id_find->second.instance_mask_,
                       merged_id_find->second.instance_mask_);
      }
    }
  }

//  //insert the lost label masks
//  for (auto dst_mask = dstSeg.instance_id_mask.begin();
//      dst_mask != dstSeg.instance_id_mask.end() ; ++dst_mask){
//    const int &dst_label = dst_mask->first;
//    if (dst_label == 0) continue;
//    //if this label is lost
//    if (mergedSeg.instance_id_mask.find(dst_label)
//        == mergedSeg.instance_id_mask.end())
//    {
//      mergedSeg.instance_id_mask.insert(*dst_mask);
//      int class_id = dstSeg.find_classID_from_instaneID(dst_label, true);
//      All_Prob_Vect class_prob = dstSeg.find_classProb_from_instaneID(dst_label, true);
//      mergedSeg.instance_class_id.insert(std::make_pair(dst_label, class_id));
//      mergedSeg.instance_id_all_prob_.insert(std::make_pair(dst_label, class_prob));
//    }
//  }

  //generate combined label masks
  mergedSeg.generate_bgLabels();
  return mergedSeg;
}


bool Segmentation::local2global(SegmentationResult& mergedSeg,
                                SegmentationResult& newModel,
                                const SegmentationResult& mask,
                                const SegmentationResult& model,
                                const int min_mask_size,
                                const float combine_threshold,
                                const float new_model_threshold){
  bool hasnewmodel = false;

  //in case that model is whole backgounrd
  if ((model.pair_instance_seg_.size()==1)&&
      (model.pair_instance_seg_.find(0) != model.pair_instance_seg_.end()) &&
      (mask.pair_instance_seg_.size()>1)){
    std::cout<<"objects list only has bg, now find new objects"<<std::endl;

//    newModel.reset();

    //now remove human from masklist (background already removed)
    //remove small objects and merge with background
    this->remove_human_and_small(mergedSeg, mask, min_mask_size);

    if (mergedSeg.pair_instance_seg_.find(INVALID) != mergedSeg.pair_instance_seg_.end()){
      if (mergedSeg.pair_instance_seg_.size()>2) {
        hasnewmodel = true;
        newModel = mergedSeg ;
      }
    }
    else{
      if (mergedSeg.pair_instance_seg_.size()>1) {
        hasnewmodel = true;
        newModel = mergedSeg ;
      }
    }

    mergedSeg.generate_bgLabels();
    return hasnewmodel;
  }

  //in case that geo2mask and volume2mask both exist
  mergedSeg = model;
  mergedSeg.generate_bgLabels(); //fill in the holes from last person position

  //number of the model masks that have been associated
  int mask2model_merged = 0;
  std::set<int> merged;

  for (auto mask_instance = mask.pair_instance_seg_.begin();
      mask_instance != mask.pair_instance_seg_.end(); ++mask_instance) {
//    float max_overlap = 0;
    const int& mask_instance_id = mask_instance->first;
    const int& mask_class_id = mask_instance->second.class_id_;
    const All_Prob_Vect& mask_class_prob = mask_instance->second.all_prob_;

    if (mask_class_id == 0) continue;
    cv::Mat mask_instance_mask =  mask_instance->second.instance_mask_;

//    ////remove human from merged lists
    if (mask_instance_id == INVALID){
      mergedSeg.pair_instance_seg_.insert(*mask_instance);
      continue;
    }

    if (human_classid_.find(mask_class_id) != human_classid_.end()){
      if (mergedSeg.pair_instance_seg_.find(INVALID)
          == mergedSeg.pair_instance_seg_.end()){
        instance_seg human(255, mask_instance_mask);
        mergedSeg.pair_instance_seg_.insert(std::make_pair(INVALID, human));
      }
      else{
        instance_seg& human_seg = mergedSeg.pair_instance_seg_.at(INVALID);
        cv::bitwise_or(mask_instance_mask, human_seg.instance_mask_, human_seg.instance_mask_);
      }
      continue;
    }

    const int mask_size = cv::countNonZero(mask_instance_mask);

    //number of the model masks that cannot be associated with this src mask
    int overlap_2model_toosmall = 0;

    for (auto model_instance = model.pair_instance_seg_.begin();
        model_instance != model.pair_instance_seg_.end() ; ++model_instance) {
      const int &model_instance_id = model_instance->first;
      const cv::Mat& model_instance_mask = model_instance->second.instance_mask_;
      const int &model_class_id = model_instance->second.class_id_;
      if (model_class_id == 0) continue;
//      std::cout<<"src_mask_mat: "<<src_mask_mat.size()<<std::endl;
//      std::cout<<"dst_mask_mat: "<<dst_mask_mat.size()<<std::endl;

      cv::Mat overlap_mask;
      cv::bitwise_and(mask_instance_mask, model_instance_mask, overlap_mask);
      //since model_mask usually is not complemte
      int model_size = cv::countNonZero(model_instance_mask);
      int overlap_size = cv::countNonZero(overlap_mask);
      const float model_ratio = static_cast<float>(overlap_size) /
          static_cast<float>(model_size);
      const float mask_ratio = static_cast<float>(overlap_size) /
          static_cast<float>(mask_size);
      const float ratio = fmaxf(model_ratio, mask_ratio);

      if (ratio>0.99f){
        const float inv_ratio = fmaxf(1./model_ratio, 1./mask_ratio);
        if (inv_ratio>1.5) continue;
      }

      //if overlap is greater than a threshold and has the same id, this local
      // mask is associated with this model mask,
      if ((ratio > combine_threshold) /*&& (model_class_id == mask_class_id)*/){

        instance_seg& merged_update = mergedSeg.pair_instance_seg_.at
            (model_instance_id);
        if (merged_update.rendered_ == true){
          //do not merge the same class mask, directly insert
//          std::cout<<"mask with instance id "<<model_instance_id
//                   <<", class id "<<model_class_id
//                   <<" coincident with the current object labels"<<std::endl;

          cv::Mat globalMask = merged_update.instance_mask_.clone();
          //implicily change rendered_ to be false
          merged_update = mask_instance->second;
          merged_update.setGlobalMask(globalMask);
          merged_update.hasModelMask_ = true;
        }
        else{
          //more than one masks coincident with one model class
          assert(merged_update.hasModelMask_ == true);
          std::cout<<"mask with instance id "<<model_instance_id
                   <<", class id "<<model_class_id
                   <<" also coincident with current object labels"<<std::endl;

          merged_update.merge(mask_instance->second);
        }

//        cv::Mat changed_mask;
//        cv::bitwise_not(mask_mask_mat, changed_mask);
//        cv::bitwise_and(changed_mask, mergedSeg.instance_id_mask[0],
//                        mergedSeg.instance_id_mask[0]);
        merged.insert(model_instance_id);
        mask2model_merged++;
        break;
      }
//      else
        //same clsss, yet the ratio is too low =>mainly for debuging
//        if (model_class_id == mask_class_id){
//          std::cout<<"mask with instance id "<<model_instance_id <<", class id " <<model_class_id
//                   <<" has the same class id with mask, yet the coincidence ratio "<<ratio<<" is too low"<<std::endl;
//        }


      //if overlap is too small to associate with this model mask,
      //count how many masks it has not been coincident
      if (ratio<new_model_threshold) {
        overlap_2model_toosmall++;
      }
    }

    //if this src mask is too small to associate with any other model masks,
    // generate a new segmentation mask
    if (overlap_2model_toosmall == (model.pair_instance_seg_.size() -1)){
      if (mask_size< min_mask_size) {
        std::cout<<"mask with instance id "<<mask_instance_id
                 <<", class id "<<mask_class_id
                 <<" is too small to"
                 <<" generate a new object"<<std::endl;
        instance_seg& bg_seg = mergedSeg.pair_instance_seg_.at(0);
        cv::bitwise_or(mask_instance_mask, bg_seg.instance_mask_, bg_seg.instance_mask_);
        continue;
      }
      else{
        std::cout<<"mask with instance id "<<mask_instance_id
                 <<", class id "<<mask_class_id
                 <<" cannot match any model masks"
                 <<", generate a new object"<<std::endl;
        hasnewmodel = true;

        //avoid duplicate integration
            /*
        int new_instance_id = mergedSeg.instance_id_mask.size();
        mergedSeg.instance_id_mask.insert(std::make_pair(new_instance_id,mask_mask_mat));
        mergedSeg.instance_class_id.insert(std::make_pair(new_instance_id, mask_class_id));
        mergedSeg.instance_id_all_prob_.insert(std::make_pair(new_instance_id,
                                                          mask_class_prob));
                                                          */
        int new_generate_id = newModel.pair_instance_seg_.size();
        newModel.pair_instance_seg_.insert(std::make_pair(new_generate_id,
                                                          mask_instance->second));
      }
    }
  }

  /*
   * only for debugging information
  // there are some models masks that cannot find the match in mask-rcnn
  if (mask2model_merged != (model.instance_id_mask.size() -1)){
    for (auto merged_mask = mergedSeg.instance_id_mask.begin();
         merged_mask != mergedSeg.instance_id_mask.end() ; ++merged_mask) {
      const int &merged_instance_id = merged_mask->first;
      const int &merged_class_id = model.find_classID_from_instaneID(merged_instance_id, true);
      if (merged_class_id == 0) continue;
      if ( merged.find(merged_instance_id) != merged.end()) continue;
        std::cout<<"mask with instance id "<<merged_instance_id
                 <<", class id "<<merged_class_id
                 <<" directly projected into labelImg"<<std::endl;
      }
    }
*/
  mergedSeg.generate_bgLabels();
  mergedSeg.label_number_ = mask.label_number_;
  newModel.label_number_ = mask.label_number_;
  return hasnewmodel;
}

/* remove human and small objects,
 * humans are labeled as INVALID and ignored in the reconstruction
 * small objects are merged with background mask
 */

void Segmentation::remove_human_and_small(SegmentationResult& output,
                                          const SegmentationResult& input,
                                          const int min_mask_size){
  output.reset();

  output.pair_instance_seg_.insert(std::make_pair(0, input.pair_instance_seg_.at(0)));

  for(auto input_instance = input.pair_instance_seg_.begin();
      input_instance != input.pair_instance_seg_.end(); input_instance++){
    const int input_instance_id = input_instance->first;
    const int input_class_id = input_instance->second.class_id_;
    cv::Mat input_mask =  input_instance->second.instance_mask_;

    if (input_class_id == 0) {
//      output.pair_instance_seg_.insert(*input_instance);
      continue;
    }

    const All_Prob_Vect& class_prob = input_instance->second.all_prob_;

    //if class id is human, put it into invalid group
    if (human_classid_.find(input_class_id) != human_classid_.end()) {
      if (output.pair_instance_seg_.find(INVALID) == output.pair_instance_seg_.end()){
        instance_seg human(255, input_mask);
        output.pair_instance_seg_.insert(std::make_pair(INVALID, human));
      }
      else{
        instance_seg& human_seg = output.pair_instance_seg_.at(INVALID);
        cv::bitwise_or(input_mask, human_seg.instance_mask_, human_seg.instance_mask_);
      }
      continue;
    }

    if (input_instance_id == INVALID){
      output.pair_instance_seg_.insert(std::make_pair(INVALID,
                                                      input_instance->second));
    }

    //if this mask is too small
    int mask_size = cv::countNonZero(input_mask);
    if (mask_size< min_mask_size) {
      instance_seg& bg_seg = output.pair_instance_seg_.at(0);
      cv::bitwise_or(input_mask, bg_seg.instance_mask_, bg_seg.instance_mask_);
      continue;
    }

    //insert the remaining mask and order its instance id
    int new_instance_id;
    if (output.pair_instance_seg_.find(INVALID) == output.pair_instance_seg_.end()){
      new_instance_id = output.pair_instance_seg_.size();
    }
    else{
      new_instance_id = output.pair_instance_seg_.size() - 1;
    }
    output.pair_instance_seg_.insert(std::make_pair(new_instance_id,
                                                    input_instance->second));
    output.label_number_ = input.label_number_;
  }

  //ensure the masks are excluding the human area
  output.exclude_human();

}