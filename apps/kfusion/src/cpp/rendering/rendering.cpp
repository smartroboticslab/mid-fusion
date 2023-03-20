 /*
 * SPDX-FileCopyrightText: 2017-2019 Smart Robotics Lab, Imperial College London
 * SPDX-FileCopyrightText: 2017-2019 Binbin Xu
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "rendering.h"

void raycastObjectList(ObjectList& objectlist, float3* vertex, float3*normal,
                       cv::Mat& labelImg, std::set<int>& objects_in_view,
                       const Matrix4& T_w_c, const float4& k, const uint2& inputSize,
                       const float nearPlane, const float farPlane,
                       const float mu,  const bool has_integrated) {
  TICK();
  unsigned int y;
  bool should_raycast_all = (objects_in_view.size() == 0);
  const Matrix4 view = T_w_c * getInverseCameraMatrix(k);
#pragma omp parallel for shared(normal, vertex, labelImg), private(y)
  for (y = 0; y < inputSize.y; y++)
#pragma simd
    for (unsigned int x = 0; x < inputSize.x; x++) {
      float nearest_hit_t = 1.0e3;
      double high_fg_prob = 0.f;
      uint miss_count = 0; //count how many object volumes this ray has missed
      uint2 pos = make_uint2(x, y);
      for (auto object = objectlist.begin(); object != objectlist.end();
           ++object){
        const int label = (*object)->instance_label_;
        if (!should_raycast_all){
          if (objects_in_view.find(label) == objects_in_view.end()){
            //this object is not in this view, skip raycasting
            miss_count++;
            continue;
          }
        }
        const Volume<FieldType> &volume = (*object)->volume_;
        const float &step = (*object)->get_volume_step();
        const float largestep = (*object)->get_volume_step() * BLOCK_SIDE;
        const Matrix4& T_w_o = (*object)->volume_pose_;

        //warp the raycast origin and direction from world coordinate to
        const float3 w_orgin = get_translation(view);
        const float3 o_origin = inverse(T_w_o) * w_orgin;
        const float3 w_direction = rotate(view, make_float3(x, y, 1.f));
        const float3 o_direction = normalize(rotate(inverse(T_w_o), w_direction));

        //check the nearest far plane
        const float nearest_farPlane = fminf(nearest_hit_t, farPlane);

        ray_iterator<typename Volume<FieldType>::field_type> ray(volume._map_index,
                                                                 o_origin,
                                                                 o_direction,
                                                                 nearPlane,
                                                                 nearest_farPlane);

        const std::tuple<float, float, float> t = ray.next(); /* Get distance to the first intersected block */
        float t_min = std::get<0>(t);
//        if ((label != 0) && (t_min > 0.f)){
//          std::cout<<std::endl;
//        }
        const float4 o_hit = t_min > 0.f ?
                           raycast(volume, o_origin, o_direction,
                                   t_min*volume._size/volume._resol,
                                   nearest_farPlane, mu, step, largestep)
                                         : make_float4(0.f);

//        if ((label != 0) && (o_hit.w >0)){
//          std::cout<<nearest_hit_t<<farPlane<<std::endl;
//          std::cout<<o_hit.w<<std::endl;
//        }

        //prior to raycast foreground first
        float hit_distance;
        if(label == 0){
          hit_distance = o_hit.w + 0.05f;
        }
        else{
          hit_distance = o_hit.w;
        }
        // find the nearest hitting point
        if (hit_distance > (nearest_hit_t)){
          miss_count++;
          continue;
        }

        //if background layer, skip it
        const double fg_prob = volume.interp(make_float3(o_hit), [](const auto&
        val){ return val.fg;});

        if (fg_prob <= 0.5f) {
          miss_count++;
          continue;
        }

        //if the hit distance same,
        // choose the one has higher foreground possibility
        if (hit_distance == nearest_hit_t){
          if (fg_prob > high_fg_prob) {
            high_fg_prob = fg_prob;
            labelImg.at<int>(y,x) = label;
            vertex[pos.x + pos.y * inputSize.x] = T_w_o * make_float3(o_hit);
            float3 o_surfNorm = volume.grad(make_float3(o_hit),
                                            [](const auto& val){ return val.x; });
            if (length(o_surfNorm) == 0) {
              //normal[pos] = normalize(surfNorm); // APN added
              normal[pos.x + pos.y * inputSize.x].x = INVALID;
            } else {
              normal[pos.x + pos.y * inputSize.x] = normalize(rotate(T_w_o, o_surfNorm));
            }
            continue;
          }
          else{
            miss_count++;
            continue;
          }
        }

        if(hit_distance < (nearest_hit_t))  {
          nearest_hit_t = hit_distance;
          high_fg_prob = fg_prob;//even if fg_prob < high_fg_prob since hit
          // distance has higher priority
          labelImg.at<int>(y,x) = label;
          vertex[pos.x + pos.y * inputSize.x] = T_w_o * make_float3(o_hit);
          float3 o_surfNorm = volume.grad(make_float3(o_hit),
                                        [](const auto& val){ return val.x; });
          if (length(o_surfNorm) == 0) {
            //normal[pos] = normalize(surfNorm); // APN added
            normal[pos.x + pos.y * inputSize.x].x = INVALID;
          } else {
            normal[pos.x + pos.y * inputSize.x] = normalize(rotate(T_w_o, o_surfNorm));
          }
          continue;
        } else {
          miss_count++;
//          std::cerr<< "RAYCAST MISS in the class id: "<<label<<", with position:"<<
//                   pos.x << " " << pos.y <<" " << o_hit.w <<"\n";
//          std::cout<<o_hit.w<<std::endl;
//          vertex[pos.x + pos.y * inputSize.x] = make_float3(0);
//          normal[pos.x + pos.y * inputSize.x] = make_float3(INVALID, 0, 0);
//          labelImg.at<int>(y,x) = 0;
          continue;
        }
      }

      if (miss_count == objectlist.size()){
//        std::cerr<< "RAYCAST MISS in all objects at position:"<<
//                   pos.x << " " << pos.y <<"\n";
        vertex[pos.x + pos.y * inputSize.x] = make_float3(0);
        normal[pos.x + pos.y * inputSize.x] = make_float3(INVALID, 0, 0);
        labelImg.at<int>(y,x) = INVALID;
      }

      for (auto object = objectlist.begin(); object != objectlist.end();
           ++object){
        float3 obj_vertex, obj_normal;
        const int pixel_object_id = labelImg.at<int>(y,x);
        if (pixel_object_id == (*object)->instance_label_){
          obj_vertex = vertex[pos.x + pos.y * inputSize.x];
          obj_normal = normal[pos.x + pos.y * inputSize.x];
        }
        else{
          obj_vertex = make_float3(0);
          obj_normal = make_float3(INVALID, 0, 0);
        }
        if (has_integrated){
          (*object)->m_vertex[pos.x + pos.y * inputSize.x] = obj_vertex;
          (*object)->m_normal[pos.x + pos.y * inputSize.x] = obj_normal;
        }
        else{
          (*object)->m_vertex_bef_integ[pos.x + pos.y * inputSize.x] = obj_vertex;
          (*object)->m_normal_bef_integ[pos.x + pos.y * inputSize.x] = obj_normal;
        }

//        //generate objects in view list
//        if ((should_raycast_all) && (pixel_object_id != INVALID)){
//          objects_in_view.insert(labelImg.at<int>(y,x));
//        }
      }
      TOCK("raycastKernel", inputSize.x * inputSize.y);
    }
}

void render_RGBD_TrackKernel(uchar4* out, const TrackData* icp_data,
                             const TrackData* rgb_data, uint2 outSize) {
  TICK();

  unsigned int y;
#pragma omp parallel for \
        shared(out), private(y)
  for (y = 0; y < outSize.y; y++)
    for (unsigned int x = 0; x < outSize.x; x++) {
      uint pos = x + y * outSize.x;
      TrackData e_icp = icp_data[pos];
      TrackData e_rgb = rgb_data[pos];

      if ((e_icp.result == 1) && (e_rgb.result == 1)){
        float e = fabs(e_rgb.error * 255.0);
        out[pos] = make_uchar4(e, e, e, 255);  // ok   pixel difference
//         out[pos] = make_uchar4(128, 128, 128, 0);  // ok   GREY
        continue;
      }

      if (((e_icp.result < -3) && (e_icp.result > -6))
//      if (((e_icp.result == -4))
      || ((e_rgb.result < -3) && (e_rgb.result > -6))){
        out[pos] = make_uchar4(255, 0, 0, 0);      // high residual RED
        continue;
      }

      if ((e_icp.result == -2) || (e_rgb.result == -2)){
        out[pos] = make_uchar4(255, 255, 0, 0);      // no input Yellow
        continue;
      }

      if ((e_icp.result == -3) || (e_rgb.result == -3)){
        out[pos] = make_uchar4(255, 255, 0, 0);      // no w_refnormal_r Yellow
        continue;
      }

      if ((e_icp.result == -1) || (e_rgb.result == -1)){
        out[pos] = make_uchar4(0, 255, 0, 0);      // not in image Green
        continue;
      }

      if ((e_icp.result == -6) || (e_rgb.result == -6)){
        out[pos] = make_uchar4(255, 175, 175, 0);     // human: pink
        continue;
      }

      // default:
      out[pos] = make_uchar4(128, 128, 128, 0);  // ok   GREY
//          std::cout<<e_icp.result<<" "<<e_rgb.result<<std::endl;
//      break;

    }
  TOCK("renderTrackKernel", outSize.x * outSize.y);
}


void volume2MaskKernel(const ObjectList& objectlist, cv::Mat& labelImg,
                       const Matrix4&T_w_c, const float4& k, const uint2& inputSize,
                       const float nearPlane, const float farPlane,
                       const float mu) {
  TICK();
  unsigned int y;
  labelImg = cv::Mat::zeros(cv::Size(inputSize.x, inputSize.y), CV_32SC1);
  //if there is only background
  //if (objectlist.size()==1) return;
  const Matrix4 view = T_w_c * getInverseCameraMatrix(k);
  #pragma omp parallel for shared(labelImg), private(y)
  for (y = 0; y < inputSize.y; y++)
#pragma simd
    for (unsigned int x = 0; x < inputSize.x; x++) {
      float nearest_hit_t = 1.0e10;
      double high_fg_prob = 0.f;
      uint miss_count = 0; //count how many object volumes this ray has missed
      for (auto object = objectlist.begin(); object != objectlist.end();
           ++object){
//        if ((*object)->class_id_ == 0) continue;
        const Volume<FieldType> &volume = (*object)->volume_;
        const float &step = (*object)->get_volume_step();
        const float largestep = (*object)->get_volume_step() * BLOCK_SIDE;
        const int &label = (*object)->instance_label_;
        const Matrix4& T_w_o = (*object)->volume_pose_;

        //warp the raycast origin and direction from world coordinate to
        const float3 w_orgin = get_translation(view);
        const float3 o_origin = inverse(T_w_o) * w_orgin;
        const float3 w_direction = rotate(view, make_float3(x, y, 1.f));
        const float3 o_direction = normalize(rotate(inverse(T_w_o), w_direction));

        //check the nearest far plane
        const float nearest_farPlane = fminf(nearest_hit_t, farPlane);

        ray_iterator<typename Volume<FieldType>::field_type> ray(volume._map_index,
                                                                 o_origin,
                                                                 o_direction,
                                                                 nearPlane,
                                                                 nearest_farPlane);

        const std::tuple<float, float, float> t = ray.next(); /* Get distance to the first intersected block */
        float t_min = std::get<0>(t);
        const float4 o_hit = t_min > 0.f ?
                             raycast(volume, o_origin, o_direction,
                                     t_min*volume._size/volume._resol,
                                     nearest_farPlane, mu, step, largestep)
                                         : make_float4(0.f);

//prior to raycast foreground first
        float hit_distance;
        if(label == 0){
          hit_distance = o_hit.w + 0.05f;
        }
        else{
          hit_distance = o_hit.w;
        }

        // find the nearest hitting point
        if (hit_distance > (nearest_hit_t)){
          miss_count++;
          continue;
        }

        //if background layer, skip it
        const double fg_prob = volume.interp(make_float3(o_hit), [](const auto&
        val){ return val.fg;});
        if (fg_prob <= 0.5f) {
          miss_count++;
          continue;
        }

        //if the hit distance same,
        // choose the one has higher foreground possibility
        if (hit_distance == nearest_hit_t){
          if (fg_prob > high_fg_prob) {
            high_fg_prob = fg_prob;
            labelImg.at<int>(y,x) = label;
            continue;
          }
          else{
            miss_count++;
            continue;
          }
        }

        if(hit_distance < (nearest_hit_t)) {
          nearest_hit_t = hit_distance;
          high_fg_prob = fg_prob;//even if fg_prob < high_fg_prob since hit
              // distance has higher priority
          labelImg.at<int>(y,x) = label;
          continue;
        } else {
          //std::cerr<< "RAYCAST MISS "<<  pos.x << " " << pos.y <<"  " << hit.w <<"\n";
          //std::cout<<o_hit.w<<std::endl;
          miss_count++;
        }
      }
      if (miss_count == objectlist.size()){
        labelImg.at<int>(y,x) = INVALID;//invalid?
      }
      TOCK("volume2Mask", inputSize.x * inputSize.y);
    }
}


void renderNormalKernel(uchar3* out, const float3* normal, uint2 normalSize) {
  TICK();
  unsigned int y;
#pragma omp parallel for \
        shared(out), private(y)
  for (y = 0; y < normalSize.y; y++)
    for (unsigned int x = 0; x < normalSize.x; x++) {
      uint pos = (x + y * normalSize.x);
      float3 n = normal[pos];
      if (n.x == -2) {
        out[pos] = make_uchar3(0, 0, 0);
      } else {
        n = normalize(n);
        out[pos] = make_uchar3(n.x * 128 + 128, n.y * 128 + 128,
            n.z * 128 + 128);
      }
    }
  TOCK("renderNormalKernel", normalSize.x * normalSize.y);
}

void renderDepthKernel(uchar4* out, float * depth, uint2 depthSize,
    const float nearPlane, const float farPlane) {
  TICK();

  float rangeScale = 1 / (farPlane - nearPlane);

  unsigned int y;
#pragma omp parallel for \
        shared(out), private(y)
  for (y = 0; y < depthSize.y; y++) {
    int rowOffeset = y * depthSize.x;
    for (unsigned int x = 0; x < depthSize.x; x++) {

      unsigned int pos = rowOffeset + x;
      if (depth[pos] < nearPlane)
        out[pos] = make_uchar4(255, 255, 255, 0); // The forth value is a padding in order to align memory
      else {
        if (depth[pos] > farPlane)
          out[pos] = make_uchar4(0, 0, 0, 0); // The forth value is a padding in order to align memory
        else {
          const float d = (depth[pos] - nearPlane) * rangeScale;
          out[pos] = gs2rgb(d);
        }
      }
    }
  }
  TOCK("renderDepthKernel", depthSize.x * depthSize.y);
}

void renderTrackKernel(uchar4* out, const TrackData* data, uint2 outSize) {
  TICK();

  unsigned int y;
#pragma omp parallel for \
        shared(out), private(y)
  for (y = 0; y < outSize.y; y++)
    for (unsigned int x = 0; x < outSize.x; x++) {
      uint pos = x + y * outSize.x;
      switch (data[pos].result) {
      case 1:
        out[pos] = make_uchar4(128, 128, 128, 0);  // ok	 GREY
        break;
      case -1:
        out[pos] = make_uchar4(0, 0, 0, 0);      // no input BLACK
        break;
      case -2:
        out[pos] = make_uchar4(255, 0, 0, 0);        // not in image RED
        break;
      case -3:
        out[pos] = make_uchar4(0, 255, 0, 0);    // no correspondence GREEN
        break;
      case -4:
        out[pos] = make_uchar4(0, 0, 255, 0);        // to far away BLUE
        break;
      case -5:
        out[pos] = make_uchar4(255, 255, 0, 0);     // wrong normal YELLOW
        break;
      case -6:
        out[pos] = make_uchar4(255, 175, 175, 0);     // human
        break;
      default:
        out[pos] = make_uchar4(255, 200, 128, 0);
        break;
      }
    }
  TOCK("renderTrackKernel", outSize.x * outSize.y);
}

void renderRGBTrackKernel(uchar4* out, const TrackData* data, uint2 outSize) {
  TICK();

  /*
  float max = -1000.f;
  float min = 1000.f;
	for (unsigned int y = 0; y < outSize.y*outSize.x; y++){
    max = fmaxf(max, data[y].error);
    min = fminf(min, data[y].error);
  }
   */
//  std::cout << "Max err: " << max << ", min err: " << min << std::endl;

  unsigned int y;
#pragma omp parallel for \
        shared(out), private(y)
  for (y = 0; y < outSize.y; y++)
    for (unsigned int x = 0; x < outSize.x; x++) {
      uint pos = x + y * outSize.x;
      float e = data[pos].error;
      // std::cout << "error pix (" << x << ", " << y << "): " << e << std::endl;
      switch (data[pos].result) {
        case 1:
          //e = data[pos].result < 0 ? 0.f : (e - min)/(max - min);
          //e = 255 * (e - min)/(max - min);
          e = fabs(e * 255.0);
          out[pos] = make_uchar4(e, e, e, 255);  // ok	 pixel difference
         // out[pos] = make_uchar4(128, 128, 128, 0);  // ok	 GREY
          break;
        case -1:
          out[pos] = make_uchar4(0, 255, 0, 0);      // no input(reference) depth Green
          break;
        case -2:
          out[pos] = make_uchar4(0, 0, 0, 0);        // not in image black
          break;
        case -3:
          out[pos] = make_uchar4(255, 0, 0, 0);    //  occluded RED
          break;
        case -4:
          out[pos] = make_uchar4(0, 0, 255, 0);        // high residual BLUE
          break;
        case -5:
          out[pos] = make_uchar4(255, 255, 0, 0);     // textureless YELLOW
          break;
        default:
          out[pos] = make_uchar4(255, 128, 128, 0);
          break;
      }



    }
  TOCK("renderTrackKernel", outSize.x * outSize.y);
}

void renderVolume_many_Kernel(const ObjectList& objectlist, uchar4* out,
                              const uint2 depthSize, const Matrix4& T_w_c,
                              const float4& k, const float nearPlane,
                              const float farPlane, const float mu,
                              const float largestep, const float3 ambient,
                              bool doraycast, bool renderColor,
                              const float3 *vertex, const float3 * normal,
                              const cv::Mat& labelImg,
                              const std::vector<uchar4> &colors) {
  TICK();
  unsigned int y;
  const float3 w_orgin = get_translation(T_w_c);
  const Matrix4 view = T_w_c * getInverseCameraMatrix(k);
//  if (doraycast) std::cout<<"raycast now to render volume"<<std::endl;
//  else std::cout<<"use last time raycasting to render volume"<<std::endl;
#pragma omp parallel for shared(out), private(y)
  for (y = 0; y < depthSize.y; y++) {
    for (unsigned int x = 0; x < depthSize.x; x++) {
      float3 o_hit = make_float3(0.0f);
      float3 w_hit = make_float3(0.0f);
      float3 w_surfNorm = make_float3(0.0f);
      float3 col = make_float3(0.0f);
      int w_label = 0;
      if(doraycast) {
        float nearest_hit_t = 1.0e10;
        double high_fg_prob = 0.f;
        uint miss_count = 0; //count how many object volumes this ray has missed
        for (auto object = objectlist.begin(); object != objectlist.end();
             ++object) {
          const Volume<FieldType> &volume = (*object)->volume_;
          const float &step = (*object)->get_volume_step();
        const int &label = (*object)->instance_label_;
          const Matrix4 &T_w_o = (*object)->volume_pose_;
          //warp the raycast origin and direction from world coordinate to
          const float3 o_origin = inverse(T_w_o) * w_orgin;
          const float3 w_direction = rotate(view, make_float3(x, y, 1.f));
          const float3 o_direction = normalize(rotate(inverse(T_w_o),
                                                      w_direction));
          //check the nearest far plane
          const float nearest_farPlane = fminf(nearest_hit_t, farPlane);

          ray_iterator<typename Volume<FieldType>::field_type>
              ray(volume._map_index,
                  o_origin,
                  (o_direction),
                  nearPlane,
                  nearest_farPlane);
          const float t_min =
              std::get<0>(ray.next()); /* Get distance to the first intersected block */
          const float4 hit = t_min > 0.f ?
                             raycast(volume, o_origin, o_direction,
                                     t_min * volume._size / volume._resol,
                                     nearest_farPlane, mu, step, largestep)
                                         : make_float4(0.f);

          //prior to raycast foreground first
          float hit_distance;
          if(label == 0){
            hit_distance = hit.w + 0.05f;
          }
          else{
            hit_distance = hit.w;
          }

          // find the nearest hitting point
          if (hit_distance > (nearest_hit_t)) {
            miss_count++;
            continue;
          }

          //if background layer, skip it
          const double fg_prob = volume.interp(make_float3(hit), [](const auto&
          val){ return val.fg;});
          if (fg_prob <= 0.5f) {
            miss_count++;
            continue;
          }

          //if the hit distance same,
          // choose the one has higher foreground possibility
          if (hit_distance == nearest_hit_t){
            if (fg_prob > high_fg_prob) {
              high_fg_prob = fg_prob;
            }
            else{
              miss_count++;
              continue;
            }
          }

          if (hit.w > 0) {
            nearest_hit_t = hit_distance;
            o_hit = make_float3(hit);
            const float3 o_surfNorm =
                volume.grad(o_hit, [](const auto &val) { return val.x; });
            if (length(o_surfNorm) > 0) {
              w_surfNorm = rotate(T_w_o, o_surfNorm);
              w_hit = T_w_o * o_hit;
              w_label = (*object)->instance_label_;
            }
          }
          else {
            miss_count++;
          }
        }
        //missed all the volumes
        if (miss_count == objectlist.size()){
//        std::cerr<< "RAYCAST MISS in all objects at position:"<<
//                   pos.x << " " << pos.y <<"\n";
          out[x + depthSize.x*y] = make_uchar4(0, 0, 0, 0); // The forth value is a padding to align memory
          continue;
        }
      }

        // not raycasting, reading the information in last raycasting
      else{
        w_hit = vertex[x + depthSize.x*y];
        w_surfNorm = normal[x + depthSize.x*y];
        w_label = labelImg.at<int>(y,x);
      }

      for (auto object = objectlist.begin(); object != objectlist.end();
           ++object) {
        if (w_label == (*object)->instance_label_) {
          const Volume<FieldType> &render_volume = (*object)->volume_;
          const Matrix4 &T_w_o = (*object)->volume_pose_;
          o_hit= inverse(T_w_o) * w_hit;
//          float3 o_surfNorm = rotate(inverse(T_w_o), w_surfNorm);

          const float3 diff = (std::is_same<FieldType, SDF>::value ?
                               normalize(w_orgin - w_hit) : normalize(w_hit - w_orgin));
          const float dir = fmaxf(dot(normalize(w_surfNorm), diff), 0.f);
          if (renderColor) {
            const float interpolated_r = render_volume.interp(o_hit,
                [](const auto &val) { return val.r; });
            const float interpolated_g = render_volume.interp(o_hit,
                [](const auto &val) { return val.g; });
            const float interpolated_b = render_volume.interp(o_hit,
                [](const auto &val) { return val.b; });
            const float3 rgb = make_float3(interpolated_r, interpolated_g,
                                           interpolated_b);
            col = clamp(make_float3(dir) * rgb + ambient, 0.f, 1.f) * 255;
          } else {
            if (w_label == 0){
              col = clamp(make_float3(dir) + ambient, 0.f, 1.f) * 255;
            }
            else{
              const float3 rgb = make_float3(colors[w_label])/255;
              col = clamp(make_float3(dir) * rgb + ambient, 0.f, 1.f) * 255;
            }
          }
        }
      }
      out[x + depthSize.x*y] = make_uchar4(col.x, col.y, col.z, 0); // The forth value is a padding to align memory
    }
  }
  TOCK("renderVolumeKernel", depthSize.x * depthSize.y);
}

void renderIntensityKernel(uchar4* out, float * intensity, uint2 framesize) {
  TICK();

  unsigned int y;
#pragma omp parallel for \
        shared(out), private(y)
  for (y = 0; y < framesize.y; y++) {
    int rowOffeset = y * framesize.x;
    for (unsigned int x = 0; x < framesize.x; x++) {
      unsigned int pos = rowOffeset + x;
      const float gs = 255.f * intensity[pos];
//      std::cout << "Pos: " << pos << " value: " << intensity[pos] << std::endl;
      out[pos] = make_uchar4(gs, gs, gs, 255);
    }
  }
  TOCK("renderIntensityKernel", framesize.x * framesize.y);
}

std::vector<uchar4> random_color(int class_nums){
  std::vector<uchar4> colors;
  srand(time(NULL));
  for (int i = 0; i < class_nums; ++i) {
    int r = static_cast<int>(((double) rand() / (RAND_MAX)) * 255);
    int g = static_cast<int>(((double) rand() / (RAND_MAX)) * 255);
    int b = static_cast<int>(((double) rand() / (RAND_MAX)) * 255);
    colors.push_back(make_uchar4(max(r, 0), max(g, 0), max(b, 0), 0));
  }
  return colors;
}


void renderMaskWithImageKernel(uchar4* out, uint2 framesize,
                               const float3* input,
                               const SegmentationResult& mask,
                               const std::vector<uchar4> &colors) {
  TICK();

  const cv::Mat& labelImg = mask.labelImg;
  unsigned int y;
  float alpha = 0.3;
#pragma omp parallel for \
        shared(out), private(y)
  for (y = 0; y < framesize.y; y++) {
    int rowOffeset = y * framesize.x;
    for (unsigned int x = 0; x < framesize.x; x++) {
      unsigned int pos = rowOffeset + x;
      const float3 rgb = input[pos] * 255.0;
      //segmented
      int instance_id = labelImg.at<int>(y,x);
      const uchar4 label_color = colors[instance_id];

      uchar r, g, b;
      if (instance_id == INVALID){
        r = rgb.x * alpha + 128 * (1-alpha);
        g = rgb.y * alpha + 128 * (1-alpha);
        b = rgb.z * 0 + 255 * (1);
        out[pos] = (make_uchar4(r, g, b, 0));
        continue;
      }

      if (instance_id != 0){
      r = rgb.x * alpha + label_color.x * (1-alpha);
      g = rgb.y * alpha + label_color.y * (1-alpha);
      b = rgb.z * alpha + label_color.z * (1-alpha);
    }
    else{
      r = rgb.x;
      g = rgb.y;
      b = rgb.z;
    }
      out[pos] = (make_uchar4(r, g, b, 0));
    }
  }
  TOCK("renderMaskKernel", framesize.x * framesize.y);
}


void renderMaskMotionWithImageKernel(uchar4* out, uint2 framesize,
                                     const float3* input,
                                     const SegmentationResult& mask,
                                     const ObjectList& objectlist,
                                     const std::vector<uchar4> &colors) {
  TICK();

  const cv::Mat& labelImg = mask.labelImg;
  unsigned int y;
  float alpha = 0.3;
#pragma omp parallel for \
        shared(out), private(y)
  for (y = 0; y < framesize.y; y++) {
    int rowOffeset = y * framesize.x;
    for (unsigned int x = 0; x < framesize.x; x++) {
      unsigned int pos = rowOffeset + x;
      const float3 rgb = input[pos] * 255.0;
      //segmented
      int instance_id = labelImg.at<int>(y,x);
      const uint rgb_pos = pos + framesize.x * framesize.y;
      const uchar4 label_color = colors[instance_id];
      uchar r, g, b;
      if (instance_id == INVALID){
        r = rgb.x * alpha + 255 * (1-alpha);
        g = rgb.y * alpha + 175 * (1-alpha);
        b = rgb.z * alpha + 175 * (1-alpha);
//        out[pos] = (make_uchar4(255, 255, 0, 0));
        out[pos] = (make_uchar4(r, g, b, 0));
        continue;
      }

      if (instance_id == 0){
//        if ((objectlist.at(instance_id)->trackresult_[pos].result == -4) /*||
//            (objectlist.at(instance_id)->trackresult_[rgb_pos].result<-3)*/){
          if ((objectlist.at(instance_id)->trackresult_[pos].result < -3) ||
            (objectlist.at(instance_id)->trackresult_[rgb_pos].result<-3)){
//          r = rgb.x * alpha + label_color.x * (1-alpha);
//          g = rgb.y * alpha + label_color.y * (1-alpha);
//          b = rgb.z * alpha + label_color.z * (1-alpha);
            r = rgb.x * alpha + 255 * (1-alpha);
            g = rgb.y * alpha + 0 * (1-alpha);
            b = rgb.z * alpha + 0 * (1-alpha);
        }
        else{
          r = rgb.x;
          g = rgb.y;
          b = rgb.z;
        }
        out[pos] = (make_uchar4(r, g, b, 0));
        continue;
      }

      if ((objectlist.at(instance_id)->trackresult_[pos].result<-3) ||
          (objectlist.at(instance_id)->trackresult_[rgb_pos].result<-3)){
        r = rgb.x;
        g = rgb.y;
        b = rgb.z;
      }
      else{
        r = rgb.x * alpha + label_color.x * (1-alpha);
        g = rgb.y * alpha + label_color.y * (1-alpha);
        b = rgb.z * alpha + label_color.z * (1-alpha);

      }
      out[pos] = (make_uchar4(r, g, b, 0));
//      continue;
    }
  }
  TOCK("renderMaskMotionKernel", framesize.x * framesize.y);
}


void renderInstanceMaskKernel(uchar4* out, uint2 framesize,
                              const SegmentationResult& mask,
                              const std::vector<uchar4> &colors) {
  TICK();

  const cv::Mat& labelImg = mask.labelImg;
  unsigned int y;
#pragma omp parallel for \
        shared(out), private(y)
  for (y = 0; y < framesize.y; y++) {
    int rowOffeset = y * framesize.x;
    for (unsigned int x = 0; x < framesize.x; x++) {
      unsigned int pos = rowOffeset + x;

      //segmented out
      int instance_id = labelImg.at<int>(y,x);
    //      std::cout<<labelImg.type()<<std::endl;
//      std::cout<<labelImg.at<int>(y,x)<<std::endl;

        out[pos] = colors[instance_id];
    }
  }
  TOCK("renderIntensityKernel", framesize.x * framesize.y);
}

void renderClassMaskKernel(uchar4* out, uint2 framesize,
                           const SegmentationResult& mask,
                           const std::vector<uchar4> &colors) {
  TICK();

  const cv::Mat& labelImg = mask.labelImg;
  unsigned int y;
#pragma omp parallel for \
        shared(out), private(y)
  for (y = 0; y < framesize.y; y++) {
    int rowOffeset = y * framesize.x;
    for (unsigned int x = 0; x < framesize.x; x++) {
      unsigned int pos = rowOffeset + x;

      //segmented out
      const int& instance_id = labelImg.at<int>(y,x);
      const int &class_id = mask.pair_instance_seg_.at(instance_id).class_id_;

      //      std::cout<<labelImg.type()<<std::endl;
//      std::cout<<labelImg.at<int>(y,x)<<std::endl;

      out[pos] = colors[class_id];
    }
  }
  TOCK("renderIntensityKernel", framesize.x * framesize.y);
}

void renderMaskKernel(uchar4* out, float * intensity, uint2 framesize, cv::Mat labelImg, std::vector<uchar4> colors) {
  TICK();

  unsigned int y;
#pragma omp parallel for \
        shared(out), private(y)
  for (y = 0; y < framesize.y; y++) {
    int rowOffeset = y * framesize.x;
    for (unsigned int x = 0; x < framesize.x; x++) {
      unsigned int pos = rowOffeset + x;

      //segmented out
      int label = labelImg.at<int>(y,x);
//      std::cout<<labelImg.type()<<std::endl;
//      std::cout<<labelImg.at<int>(y,x)<<std::endl;

      if (label!=0){
        out[pos] = colors[label];
      }
      else{
        const float gs = 255.f * intensity[pos];
        out[pos] = make_uchar4(gs, gs, gs, 255);
      }
        }
      }
  TOCK("renderIntensityKernel", framesize.x * framesize.y);
}

void renderIntensityKernel(uchar4* out, float * intensity, uint2 framesize, std::vector<cv::Mat> idx_pixel,
                           std::vector<int> class_id, std::vector<uchar4> colors) {

    TICK();

  unsigned int y;
#pragma omp parallel for \
        shared(out), private(y)
  for (y = 0; y < framesize.y; y++) {
    int rowOffeset = y * framesize.x;
    for (unsigned int x = 0; x < framesize.x; x++) {
      unsigned int pos = rowOffeset + x;
      const float gs = 255.f * intensity[pos];
//      std::cout << "Pos: " << pos << " value: " << intensity[pos] << std::endl;
      out[pos] = make_uchar4(gs, gs, gs, 255);
      if (class_id.size()>0){
        for (size_t i = 0; i < class_id.size(); ++i) {
          const cv::Mat& maskImg = idx_pixel[i];
//          int mask_pos = x* framesize.y + y;
          if (maskImg.at<uchar>(y, x) !=0){
            out[pos] = colors[i];
          }
        }
      }
    }
  }
  TOCK("renderIntensityKernel", framesize.x * framesize.y);
}


void check_static_state(cv::Mat& object_inlier_mask,
                        const cv::Mat& object_mask,
                        const TrackData* camera_tracking_result,
                        const uint2 framesize,
                        const bool use_icp,
                        const bool use_rgb){
  //object_inlier_mask = object_mask.clone();
  unsigned int y;
#pragma omp parallel for \
        shared(object_inlier_mask), private(y)
  for (y = 0; y < framesize.y; y++) {
    int rowOffeset = y * framesize.x;
    for (unsigned int x = 0; x < framesize.x; x++) {
      unsigned int pos = rowOffeset + x;
      const TrackData& icp_error = camera_tracking_result[pos];
      unsigned int rgb_pos = pos + framesize.x * framesize.y;
      const TrackData& rgb_error = camera_tracking_result[rgb_pos];

      bool is_inlier = false;
      if (use_icp && use_rgb){
        is_inlier = (icp_error.result>-3) && (rgb_error.result>-3);
      }
      else{
        if ((use_icp) && (!use_rgb)){
          is_inlier = (icp_error.result>-3);
        }
        if ((!use_icp) && (use_rgb)){
          is_inlier = (rgb_error.result>-3);
        }
      }

      if ((object_mask.at<uchar>(y,x)!=0) && is_inlier){
        object_inlier_mask.at<uchar>(y,x) = 255;
      }

      else{
        object_inlier_mask.at<uchar>(y,x) = 0;
      }
    }
  }
}


void opengl2opencv_kernel(cv::Mat& renderImg, const uchar4* data, const
uint2 outSize){
  unsigned int y;
#pragma omp parallel for \
        shared(renderImg), private(y)
  for (y = 0; y < outSize.y; y++)
    for (unsigned int x = 0; x < outSize.x; x++) {
      uint pos = x + y * outSize.x;
      renderImg.at<cv::Vec3b>(y,x)[2] = data[pos].x; //R
      renderImg.at<cv::Vec3b>(y,x)[1] = data[pos].y; //G
      renderImg.at<cv::Vec3b>(y,x)[0] = data[pos].z;  //B
//      std::cout<<data[pos].x<<data[pos].y<<data[pos].z<<std::endl;
    }
}

void opengl2opencv(const uchar4* data, const uint2 outSize, const uint frame,
                   const std::string filename){
  cv::Mat renderImg = cv::Mat::zeros(cv::Size(outSize.x, outSize.y), CV_8UC3);
  opengl2opencv_kernel(renderImg, data, outSize);
  std::ostringstream name;
  name << filename+"_"<<std::setfill('0') << std::setw(5) << std::to_string(frame)<<".png";
  cv::imwrite(name.str(), renderImg);
}
