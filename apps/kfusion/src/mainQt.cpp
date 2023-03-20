/*
 * SPDX-FileCopyrightText: 2016-2019 Emanuele Vespa
 * SPDX-FileCopyrightText: 2017-2019 Smart Robotics Lab, Imperial College London
 * SPDX-FileCopyrightText: 2017-2019 Binbin Xu
 * SPDX-License-Identifier: BSD-3-Clause
 */


#include <kernels.h>
#include <interface.h>
#include <default_parameters.h>
#include <stdint.h>
#include <tick.h>
#include <vector>
#include <sstream>
#include <string>
#include <cstring>

#include <sys/types.h>
#include <sys/stat.h>
#include <sstream>
#include <iomanip>
#include <getopt.h>
#include <perfstats.h>
#include <PowerMonitor.h>

//read mask and class from mask-rcnn
#include "segmentation.h"

#ifndef __QT__

#include <draw.h>
#endif


PerfStats Stats;
PowerMonitor *powerMonitor = NULL;
uint16_t * inputDepth = NULL;
static uchar3 * inputRGB = NULL;
static uchar4 * depthRender = NULL;
static uchar4 * raycastRender = NULL;
static uchar4 * trackRender = NULL;
static uchar4 * segmentRender = NULL;
static uchar4 * volumeRender = NULL;
static uchar4 * extraRender = NULL;
static DepthReader *reader = NULL;
static Kfusion *kfusion = NULL;

static float3 init_pose;
static std::ostream* logstream = &std::cout;
static std::ofstream logfilestream;
/*
 int          compute_size_ratio = default_compute_size_ratio;
 std::string  input_file         = "";
 std::string  log_file           = "" ;
 std::string  dump_volume_file   = "" ;
 float3       init_poseFactors   = default_initial_pos_factor;
 int          integration_rate   = default_integration_rate;
 float3       volume_size        = default_volume_size;
 uint3        volume_resolution  = default_volume_resolution;
 */
DepthReader *createReader(Configuration *config, std::string filename = "");
int processAll(DepthReader *reader, bool processFrame, bool renderImages,
    Configuration *config, bool reset = false);

void qtLinkKinectQt(int argc, char *argv[], Kfusion **_kfusion,
    DepthReader **_depthReader, Configuration *config, void *depthRender,
    void *trackRender, void *volumeModel, void *inputRGB);
void storeStats(int frame, double *timings, float3 pos, bool tracked,
    bool integrated) {
  Stats.sample("frame", frame, PerfStats::FRAME);
  Stats.sample("acquisition", timings[1] - timings[0], PerfStats::TIME);
  Stats.sample("preprocessing", timings[2] - timings[1], PerfStats::TIME);
  Stats.sample("tracking", timings[4] - timings[3], PerfStats::TIME);
  Stats.sample("segmentation", timings[3] - timings[2] +timings[5] - timings[4], PerfStats::TIME);
  Stats.sample("integration", timings[6] - timings[5], PerfStats::TIME);
  Stats.sample("raycasting", timings[7] - timings[6], PerfStats::TIME);
  Stats.sample("rendering", timings[8] - timings[7], PerfStats::TIME);
  Stats.sample("computation", timings[7] - timings[1], PerfStats::TIME);
  Stats.sample("total", timings[8] - timings[0], PerfStats::TIME);
  Stats.sample("X", pos.x, PerfStats::DISTANCE);
  Stats.sample("Y", pos.y, PerfStats::DISTANCE);
  Stats.sample("Z", pos.z, PerfStats::DISTANCE);
  Stats.sample("tracked", tracked, PerfStats::INT);
  Stats.sample("integrated", integrated, PerfStats::INT);
}




/***
 * This program loop over a scene recording
 */


int main(int argc, char ** argv) {


  Configuration config = parseArgs(argc, argv);
  powerMonitor = new PowerMonitor();

  // ========= READER INITIALIZATION  =========
  reader = createReader(&config);

  //  =========  BASIC PARAMETERS  (input size / computation size )  =========
  uint2 inputSize =
      (reader != NULL) ? reader->getinputSize() : make_uint2(640, 480);
  const uint2 computationSize = make_uint2(
      inputSize.x / config.compute_size_ratio,
      inputSize.y / config.compute_size_ratio);

  //  =========  BASIC BUFFERS  (input / output )  =========

  // Construction Scene reader and input buffer
  //we could allocate a more appropriate amount of memory (less) but this makes life hard if we switch up resolution later;
  inputDepth =
        (uint16_t*) malloc(sizeof(uint16_t) * inputSize.x*inputSize.y);
  inputRGB =
        (uchar3*) malloc(sizeof(uchar3) * inputSize.x*inputSize.y);
  depthRender =
        (uchar4*) malloc(sizeof(uchar4) * computationSize.x*computationSize.y);
  raycastRender =
      (uchar4*) malloc(sizeof(uchar4) * computationSize.x*computationSize.y);
  trackRender =
        (uchar4*) malloc(sizeof(uchar4) * computationSize.x*computationSize.y);
  segmentRender =
      (uchar4*) malloc(sizeof(uchar4) * computationSize.x*computationSize.y);
  volumeRender =
        (uchar4*) malloc(sizeof(uchar4) * computationSize.x*computationSize.y);
  extraRender =
      (uchar4*) malloc(sizeof(uchar4) * computationSize.x*computationSize.y);

  init_pose = config.initial_pos_factor * config.volume_size;
  kfusion = new Kfusion(computationSize, config.volume_resolution,
      config.volume_size, init_pose, config.pyramid, config);
  kfusion->setPoseScale(config.initial_pos_factor);

  if (config.log_file != "") {
    logfilestream.open(config.log_file.c_str());
    logstream = &logfilestream;
  }

  logstream->setf(std::ios::fixed, std::ios::floatfield);

  //temporary fix to test rendering fullsize
  config.render_volume_fullsize = false;

  //The following runs the process loop for processing all the frames, if QT is specified use that, else use GLUT
  //We can opt to not run the gui which would be faster
  if (!config.no_gui) {
#ifdef __QT__
    qtLinkKinectQt(argc,argv, &kfusion, &reader, &config, depthRender, trackRender, volumeRender, inputRGB);
#else
    if ((reader == NULL) || (reader->cameraActive == false)) {
      std::cerr << "No valid input file specified\n";
      exit(1);
    }



    while (processAll(reader, true, true, &config, false) == 0) {
      drawthem(extraRender, depthRender, trackRender, volumeRender,
          segmentRender, raycastRender, computationSize, computationSize,
                    computationSize, computationSize, computationSize, computationSize);
//      if (reader->getFrameNumber() >250) {
      if (config.pause) getchar();
//      }
    }
#endif
  } else {
    if ((reader == NULL) || (reader->cameraActive == false)) {
      std::cerr << "No valid input file specified\n";
      exit(1);
    }
    while (processAll(reader, true, true, &config, false) == 0) {
    }
    std::cout << __LINE__ << std::endl;
  }
  // ==========     DUMP VOLUME      =========

  if (config.dump_volume_file != "") {
    double start = tock();
    kfusion->dump_mesh(config.dump_volume_file);
    double end = tock();
    Stats.sample("meshing", end - start, PerfStats::TIME);
  }

  //if (config.log_file != "") {
  //	std::ofstream logStream(config.log_file.c_str());
  //	Stats.print_all_data(logStream);
  //	logStream.close();
  //}
  //
  if (powerMonitor && powerMonitor->isActive()) {
    std::ofstream powerStream("power.rpt");
    powerMonitor->powerStats.print_all_data(powerStream);
    powerStream.close();
  }
  std::cout << "{";
  //powerMonitor->powerStats.print_all_data(std::cout, false);
  //std::cout << ",";
  Stats.print_all_data(std::cout, false);
  std::cout << "}" << std::endl;

  //  =========  FREE BASIC BUFFERS  =========

  free(inputDepth);
  free(depthRender);
  free(trackRender);
  free(volumeRender);
  free(extraRender);

}

//void loadMask(){
//
//}

int processAll(DepthReader *reader, bool processFrame, bool renderImages,
    Configuration *config, bool reset) {
  static float duration = tick();
  static int frameOffset = 0;
  static bool firstFrame = true;
  bool tracked, integrated, raycasted, segmented, hasmaskrcnn;
  double start, end, startCompute, endCompute;
  uint2 render_vol_size;
  double timings[9];
  float3 pos;
  int frame;
  const uint2 inputSize =
      (reader != NULL) ? reader->getinputSize() : make_uint2(640, 480);
  float4 camera =
      (reader != NULL) ?
          (reader->getK() / config->compute_size_ratio) :
          make_float4(0.0);
  if (config->camera_overrided)
    camera = config->camera / config->compute_size_ratio;

  if (reset) {
    frameOffset = reader->getFrameNumber();
  }
  bool finished = false;

  if (processFrame) {
    Stats.start();
  }
  Matrix4 pose;
  timings[0] = tock();
  if (processFrame && (reader->readNextDepthFrame(inputRGB, inputDepth)) &&
      (kfusion->MaskRCNN_next_frame(reader->getFrameNumber() - frameOffset,
                                    config->maskrcnn_folder))) {
    frame = reader->getFrameNumber() - frameOffset;
   if (frame<kfusion->segment_startFrame_) {
         std::cout<<frame<<" finished"<<std::endl;
        return finished;
    }
    if (powerMonitor != NULL && !firstFrame)
      powerMonitor->start();

    timings[1] = tock();

    if (kfusion->render_color_) {
      kfusion->preprocessing(inputDepth, inputRGB, inputSize, config->bilateralFilter);
    } else{
      kfusion->preprocessing(inputDepth, inputSize, config->bilateralFilter);
    }

    timings[2] = tock();

    hasmaskrcnn = kfusion->readMaskRCNN(camera, frame, config->maskrcnn_folder);
    timings[3] = tock();

    tracked = kfusion->tracking(camera, config->tracking_rate, frame);

    pos = kfusion->getPosition();
    pose = kfusion->getPose();

    timings[4] = tock();

    segmented = kfusion->segment(camera, frame, config->maskrcnn_folder, hasmaskrcnn);

    timings[5] = tock();

    integrated = kfusion->integration(camera, config->integration_rate,
        config->mu, frame);

    timings[6] = tock();

    raycasted = kfusion->raycasting(camera, config->mu, frame);

    timings[7] = tock();


  } else {
    if (processFrame) {
      finished = true;
      timings[0] = tock();
    }

  }
  if (renderImages) {
    int render_frame = (processFrame ? reader->getFrameNumber() - frameOffset : 0);

    /////////////visualize the segmentation/////////


/*
    kfusion->renderInstance(depthRender, kfusion->getComputationResolution(),
                         kfusion->mask_rcnn);
    kfusion->renderInstance(trackRender, kfusion->getComputationResolution(),
                            (kfusion->geo_mask));
    kfusion->renderInstance(volumeRender, kfusion->getComputationResolution(),
                            (kfusion->geo2mask_result));
    kfusion->renderInstance(raycastRender, kfusion->getComputationResolution(),
                            (kfusion->rendered_mask_));
    kfusion->renderInstance(segmentRender, kfusion->getComputationResolution(),
                         *(kfusion->frame_masks_));
*/

    kfusion->renderTrack(trackRender, kfusion->getComputationResolution(), 2,
                         frame);
//    kfusion->renderMaskWithImage(trackRender, kfusion->getComputationResolution(),
//                                 *(kfusion->frame_masks_), render_frame, "instance_geom");

    kfusion->renderMaskMotionWithImage(segmentRender,
                                       kfusion->getComputationResolution(),
                                       *(kfusion->frame_masks_), render_frame);
    kfusion->renderVolume(volumeRender, kfusion->getComputationResolution(),
                          (processFrame ? reader->getFrameNumber() - frameOffset : 0),
                          config->rendering_rate, camera, 0.75 * config->mu,
                          false);
    kfusion->renderVolume(extraRender, kfusion->getComputationResolution(),
                          render_frame,
                          config->rendering_rate, camera, 0.75 * config->mu,
                          true);
    kfusion->renderInstance(raycastRender, kfusion->getComputationResolution(),
                            (kfusion->rendered_mask_), render_frame, "rendered_mask_");

    kfusion->renderMaskWithImage(depthRender, kfusion->getComputationResolution(),
                                 (kfusion->geo2mask_result), render_frame, "geo2mask_result");

    ////////visualize the whole process/////////
////




/*
 * //    kfusion->renderDepth(depthRender,  kfusion->getComputationResolution());
//    kfusion->renderDepth(depthRender,  kfusion->getComputationResolution(), render_frame);
//    kfusion->save_input(inputRGB, inputSize, render_frame);

    kfusion->renderTrack(depthRender, kfusion->getComputationResolution(), 2,
                         frame);

//    kfusion->renderTrack(depthRender, kfusion->getComputationResolution(), 0);
//    kfusion->renderTrack(trackRender, kfusion->getComputationResolution(), 0,
//                         frame);
//    kfusion->renderInstance(raycastRender, kfusion->getComputationResolution(),
//                            (kfusion->raycast_mask_));

//    kfusion->renderInstance(segmentRender, kfusion->getComputationResolution(),
//                         *(kfusion->frame_masks_));
    kfusion->renderMaskWithImage(trackRender, kfusion->getComputationResolution(),
                                 *(kfusion->frame_masks_), render_frame, "instance_geom");

    kfusion->renderMaskMotionWithImage(segmentRender,
                                       kfusion->getComputationResolution(),
                                       *(kfusion->frame_masks_), render_frame);

    kfusion->renderVolume(volumeRender, kfusion->getComputationResolution(),
                          render_frame,
                          config->rendering_rate, camera, 0.75 * config->mu,
                          true);
    kfusion->renderVolume(raycastRender, kfusion->getComputationResolution(),
                          (processFrame ? reader->getFrameNumber() - frameOffset : 0),
                          config->rendering_rate, camera, 0.75 * config->mu,
                          false);
                          */



    timings[8] = tock();
    std::cout<<frame<<" finished"<<std::endl;
  }

  if (!finished) {
    if (powerMonitor != NULL && !firstFrame)
      powerMonitor->sample();

    float xt = pose.data[0].w - init_pose.x;
    float yt = pose.data[1].w - init_pose.y;
    float zt = pose.data[2].w - init_pose.z;
    storeStats(frame, timings, pos, tracked, integrated);
    if((config->no_gui) || (config->log_file != "")){
      *logstream << reader->getFrameNumber() << "\t" << xt << "\t" << yt << "\t" << zt << "\t" << std::endl;
    }

    if (config->log_file != ""){
      kfusion->save_poses(config->log_file, reader->getFrameNumber());
      kfusion->save_times(config->log_file, reader->getFrameNumber(), timings);
    }

    //if (config->no_gui && (config->log_file == ""))
    //	Stats.print();
    firstFrame = false;
  }
  return (finished);
}

