/*
 * SPDX-FileCopyrightText: 2016-2019 Emanuele Vespa
 * SPDX-FileCopyrightText: 2017-2019 Smart Robotics Lab, Imperial College London
 * SPDX-FileCopyrightText: 2017-2019 Binbin Xu
 * SPDX-License-Identifier: BSD-3-Clause
*/

#ifndef DEFAULT_PARAMETERS_H_
#define DEFAULT_PARAMETERS_H_

#include <vector_types.h>
#include <cutil_math.h>
#include <vector>
#include <sstream>
#include <getopt.h>

#include <constant_parameters.h>
#include <config.h>

////////////////////////// RUNTIME PARAMETERS //////////////////////

#define DEFAULT_ITERATION_COUNT 3
static const int default_iterations[DEFAULT_ITERATION_COUNT] = { 10, 5, 4 };

const float default_mu = 0.1f;
const bool default_blocking_read = false;
const int default_fps = 0;
const float default_icp_threshold = 1e-5;
const int default_compute_size_ratio = 1;
const int default_integration_rate = 2;
const int default_rendering_rate = 4;
const int default_tracking_rate = 1;
const uint3 default_volume_resolution = make_uint3(256, 256, 256);
const float3 default_volume_size = make_float3(2.f, 2.f, 2.f);
const float3 default_initial_pos_factor = make_float3(0.5f, 0.5f, 0.0f);
const bool default_no_gui = false;
const bool default_render_volume_fullsize = false;
const bool default_bilateralFilter = false;
const std::string default_dump_volume_file = "";
const std::string default_input_file = "";
const std::string default_log_file = "";
const int default_voxel_block_size = 8;
const int default_color_integration = false;
const int default_multi_resolution = false;
const bool default_bayesian = false;
const std::string default_groundtruth_file = "";
const std::string default_maskrcnn_folder = "";
const std::string default_gt_mask_folder = "";
const bool default_in_debug = false;
const bool default_pause = false;
const std::string default_output_images = "";
const float default_min_obj_ratio = 0.01;
const bool default_disable_rgb = false;
const bool default_absorb_outlier_bg = false;
const bool default_geom_refine_human = false;
const float default_obj_moved = 0.9;
const int default_init_frame = 0;

inline std::string pyramid2str(std::vector<int> v) {
	std::ostringstream ss;
	for (std::vector<int>::iterator it = v.begin(); it != v.end(); it++)
		ss << *it << " ";
	return ss.str();

}

static std::string short_options =
    "a:qc:d:f:g:hi:l:m:k:o:p:r:s:t:v:y:z:B:FC:M:S:R:D:P:O:G:T:A:H:V:I";

static struct option long_options[] =
  {
		    {"quaternion",  				   required_argument, 0, 'a'},
			  {"block-read",  				   no_argument, 0, 'b'},
		    {"compute-size-ratio",     required_argument, 0, 'c'},
		    {"dump-volume",  		   required_argument, 0, 'd'},
			  {"invert-y",  		   			no_argument, 0, 'e'},
		    {"fps",  				   required_argument, 0, 'f'},
		    {"input-file",  		   required_argument, 0, 'i'},
		    {"camera",  			   required_argument, 0, 'k'},
		    {"icp-threshold", 	 	   required_argument, 0, 'l'},
		    {"log-file",  			   required_argument, 0, 'o'},
		    {"mu", 			 		   required_argument, 0, 'm'},
		    {"init-pose",  			   required_argument, 0, 'p'},
		    {"no-gui",  			   no_argument,       0, 'q'},
		    {"integration-rate",  	   required_argument, 0, 'r'},
		    {"volume-size",  		   required_argument, 0, 's'},
		    {"tracking-rate", 		   required_argument, 0, 't'},
		    {"volume-resolution",      required_argument, 0, 'v'},
		    {"pyramid-levels", 		   required_argument, 0, 'y'},
		    {"rendering-rate",         required_argument, 0, 'z'},
        {"voxel-block-size",       required_argument, 0, 'B'},
        {"bilateral-filter",       no_argument, 0, 'F'},
        {"colour-voxels",       no_argument, 0, 'C'},
        {"multi-res",       no_argument, 0, 'M'},
        {"bayesian",       no_argument, 0, 'h'},
        {"ground-truth",       required_argument, 0, 'g'},
        {"mask-rcnn-folder",       required_argument, 0, 'S'},
        {"ground-truth-segment",       required_argument, 0, 'R'},
        {"in-debug",  			   no_argument,       0, 'D'},
        {"pause",  			   no_argument,       0, 'P'},
        {"output-images",  			   required_argument,       0, 'O'},
        {"min-obj-ratio",          required_argument, 0,'G'},
        {"disable-rgb", no_argument, 0, 'T'},
        {"absorb-outlier-bg", no_argument, 0, 'A'},
        {"geom_refine_human", no_argument, 0, 'H'},
        {"obj-moved", required_argument, 0, 'V'},
        {"init", required_argument, 0, 'I'},
		    {0, 0, 0, 0}

};

inline
void print_arguments() {
  std::cerr << "-B  (--voxel_block_size)         : default is " << default_voxel_block_size << std::endl;
  std::cerr << "-b  (--block-read)       		: default is False: Block on read " << std::endl;
  std::cerr << "-c  (--compute-size-ratio)       : default is " << default_compute_size_ratio << "   (same size)      " << std::endl;
  std::cerr << "-e  (--invert-y)       			: default is False: Block on read " << std::endl;
  std::cerr << "-d  (--dump-volume) <filename>   : Output volume file              " << std::endl;
  std::cerr << "-f  (--fps)                      : default is " << default_fps       << std::endl;
  std::cerr << "-F  (--bilateral-filter          : default is disabled"               << std::endl;
  std::cerr << "-h  (--bayesian                  : default is disabled"               << std::endl;
  std::cerr << "-i  (--input-file) <filename>    : Input camera file               " << std::endl;
  std::cerr << "-k  (--camera)                   : default is defined by input     " << std::endl;
  std::cerr << "-l  (--icp-threshold)                : default is " << default_icp_threshold << std::endl;
  std::cerr << "-o  (--log-file) <filename>      : default is stdout               " << std::endl;
  std::cerr << "-m  (--mu)                       : default is " << default_mu << "               " << std::endl;
  std::cerr << "-p  (--init-pose)                : default is " << default_initial_pos_factor.x << "," << default_initial_pos_factor.y << "," << default_initial_pos_factor.z << "     " << std::endl;
  std::cerr << "-q  (--no-gui)                   : default is to display gui"<<std::endl;
  std::cerr << "-r  (--integration-rate)         : default is " << default_integration_rate << "     " << std::endl;
  std::cerr << "-s  (--volume-size)              : default is " << default_volume_size.x << "," << default_volume_size.y << "," << default_volume_size.z << "      " << std::endl;
  std::cerr << "-t  (--tracking-rate)            : default is " << default_tracking_rate << "     " << std::endl;
  std::cerr << "-v  (--volume-resolution)        : default is " << default_volume_resolution.x << "," << default_volume_resolution.y << "," << default_volume_resolution.z << "    " << std::endl;
  std::cerr << "-y  (--pyramid-levels)           : default is 10,5,4     " << std::endl;
  std::cerr << "-z  (--rendering-rate)   : default is " << default_rendering_rate << std::endl;
  std::cerr << "-g  (--ground-truth) <filename>   : ground truth trajectory file"<< std::endl;
  std::cerr << "-S  (--mask-rcnn-folder) <foldername>   : Mask-RCNN segmentation folder"<< std::endl;
  std::cerr << "-R  (--ground-truth-segment) <foldername>   : Groundtruth "
               "segmentation folder"<< std::endl;
  std::cerr << "-D  (--in-debug)                 : default is not in debug"<<std::endl;
  std::cerr << "-P  (--pause)                   : default is not pause"<<std::endl;
  std::cerr << "-O  (--output-images)           : default is not output images"<<std::endl;
  std::cerr << "-G  (--min-obj-ratio)           : default is 0.01"<<std::endl;
  std::cerr << "-T  (--diable-rgb)           : default is false"<<std::endl;
  std::cerr << "-A  (--absorb-outlier-bg)           : default is false"<<std::endl;
  std::cerr << "-H  (--geom_refine_human)           : default is false"<<std::endl;
  std::cerr << "-V  (--obj-moved)           : default is "<<default_obj_moved<<std::endl;
  std::cerr << "-I  (--init)           : default is " <<default_init_frame<<std::endl;

}

inline float3 atof3(char * optarg) {
  float3 res;
  std::istringstream dotargs(optarg);
  std::string s;
  if (getline(dotargs, s, ',')) {
    res.x = atof(s.c_str());
  } else
    return res;
  if (getline(dotargs, s, ',')) {
    res.y = atof(s.c_str());
  } else {
    res.y = res.x;
    res.z = res.y;
    return res;
  }
  if (getline(dotargs, s, ',')) {
    res.z = atof(s.c_str());
  } else {
    res.z = res.y;
  }
  return res;
}

inline uint3 atoi3(char * optarg) {
  uint3 res;
  std::istringstream dotargs(optarg);
  std::string s;
  if (getline(dotargs, s, ',')) {
    res.x = atoi(s.c_str());
  } else
    return res;
  if (getline(dotargs, s, ',')) {
    res.y = atoi(s.c_str());
  } else {
    res.y = res.x;
    res.z = res.y;
    return res;
  }
  if (getline(dotargs, s, ',')) {
    res.z = atoi(s.c_str());
  } else {
    res.z = res.y;
  }
  return res;
}

inline float4 atof4(char * optarg) {
  float4 res;
  std::istringstream dotargs(optarg);
  std::string s;
  if (getline(dotargs, s, ',')) {
    res.x = atof(s.c_str());
  } else
    return res;
  if (getline(dotargs, s, ',')) {
    res.y = atof(s.c_str());
  } else {
    res.y = res.x;
    res.z = res.y;
    res.w = res.z;
    return res;
  }
  if (getline(dotargs, s, ',')) {
    res.z = atof(s.c_str());
  } else {
    res.z = res.y;
    res.w = res.z;
    return res;
  }
  if (getline(dotargs, s, ',')) {
    res.w = atof(s.c_str());
  } else {
    res.w = res.z;
  }
  return res;
}

Configuration parseArgs(unsigned int argc, char ** argv) {

  Configuration config;

  config.compute_size_ratio = default_compute_size_ratio;
  config.integration_rate = default_integration_rate;
  config.tracking_rate = default_tracking_rate;
  config.rendering_rate = default_rendering_rate;
  config.volume_resolution = default_volume_resolution;
  config.volume_size = default_volume_size;
  config.initial_pos_factor = default_initial_pos_factor;
  config.voxel_block_size = default_voxel_block_size;
  //initial_pose_quant.setIdentity();
  //invert_y = false;

  config.dump_volume_file = default_dump_volume_file;
  config.input_file = default_input_file;
  config.log_file = default_log_file;
  config.groundtruth_file = default_groundtruth_file;

  config.mu = default_mu;
  config.fps = default_fps;
  config.blocking_read = default_blocking_read;
  config.icp_threshold = default_icp_threshold;
  config.no_gui = default_no_gui;
  config.render_volume_fullsize = default_render_volume_fullsize;
  config.camera_overrided = false;
  config.bilateralFilter = default_bilateralFilter;
  config.bayesian = default_bayesian;
  config.in_debug = default_in_debug;
  config.pause = default_pause;
  config.output_images = default_output_images;
  config.min_obj_ratio = default_min_obj_ratio;
  config.absorb_outlier_bg = default_absorb_outlier_bg;
  config.geom_refine_human = default_geom_refine_human;
  config.obj_moved = default_obj_moved;
  config.init_frame = default_init_frame;
  config.disable_rgb = default_disable_rgb;

  config.pyramid.clear();
  for (int i = 0; i < DEFAULT_ITERATION_COUNT; i++) {
    config.pyramid.push_back(default_iterations[i]);
  }
  config.maskrcnn_folder = default_maskrcnn_folder;
  config.gt_mask_folder = default_gt_mask_folder;

  int c;
  int option_index = 0;
  int flagErr = 0;
  while ((c = getopt_long(argc, argv, short_options.c_str(), long_options,
          &option_index)) != -1)
    switch (c) {
      case 'a':
        {
          //float4 vals = atof4(optarg);
          //initial_pose_quant = Eigen::Quaternionf(vals.w, vals.x, vals.y, vals.z);

          //std::cerr << "update quaternion rotation to " << config.initial_pose_quant.x() << ","
          //<< config.initial_pose_quant.y() << "," << config.initial_pose_quant.z() << ","
          //<< config.initial_pose_quant.w() << std::endl;
          break;
        }
      case 'b':
        config.blocking_read = true;
        std::cerr << "activate blocking read" << std::endl;
        break;
      case 'c':  //   -c  (--compute-size-ratio)
        config.compute_size_ratio = atoi(optarg);
        std::cerr << "update compute_size_ratio to "
          << config.compute_size_ratio << std::endl;
        if ((config.compute_size_ratio != 1)
            && (config.compute_size_ratio != 2)
            && (config.compute_size_ratio != 4)
            && (config.compute_size_ratio != 8)) {
          std::cerr
            << "ERROR: --compute-size-ratio (-c) must be 1, 2 ,4 or 8  (was "
            << optarg << ")\n";
          flagErr++;
        }
        break;
      case 'd':
        config.dump_volume_file = optarg;
        std::cerr << "update dump_volume_file to "
          << config.dump_volume_file << std::endl;
        break;
      case 'e':
        //config.invert_y = true;
        //std::cerr << "Inverting Y axis (ICL-NUIM Fix)" << std::endl;
        break;

      case 'f':  //   -f  (--fps)
        config.fps = atoi(optarg);
        std::cerr << "update fps to " << config.fps << std::endl;

        if (config.fps < 0) {
          std::cerr << "ERROR: --fps (-f) must be >= 0 (was "
            << optarg << ")\n";
          flagErr++;
        }
        break;
      case 'g':
        config.groundtruth_file = optarg;
        std::cerr << "using the groundtruth file: " << config.groundtruth_file
          << std::endl;
        break;
      case 'h': // -h (--bayesian)
        config.bayesian = true;
        break;
      case 'i':    //   -i  (--input-file)
        config.input_file = optarg;
        std::cerr << "update input_file to " << config.input_file
          << std::endl;
        struct stat st;
        if (stat(config.input_file.c_str(), &st) != 0) {
          std::cerr << "ERROR: --input-file (-i) does not exist (was "
            << config.input_file << ")\n";
          flagErr++;
        }
        break;
      case 'k':    //   -k  (--camera)
        config.camera = atof4(optarg);
        config.camera_overrided = true;
        std::cerr << "update camera to " << config.camera.x << ","
          << config.camera.y << "," << config.camera.z << ","
          << config.camera.w << std::endl;
        break;
      case 'o':    //   -o  (--log-file)
        config.log_file = optarg;
        std::cerr << "update log_file to " << config.log_file
          << std::endl;
        break;
      case 'l':  //   -l (--icp-threshold)
        config.icp_threshold = atof(optarg);
        std::cerr << "update icp_threshold to " << config.icp_threshold
          << std::endl;
        break;
      case 'm':   // -m  (--mu)
        config.mu = atof(optarg);
        std::cerr << "update mu to " << config.mu << std::endl;
        break;
      case 'p':    //   -p  (--init-pose)
        config.initial_pos_factor = atof3(optarg);
        std::cerr << "update init_poseFactors to "
          << config.initial_pos_factor.x << ","
          << config.initial_pos_factor.y << ","
          << config.initial_pos_factor.z << std::endl;
        break;
      case 'q':
        config.no_gui = true;
        break;
      case 'r':    //   -r  (--integration-rate)
        config.integration_rate = atoi(optarg);
        std::cerr << "update integration_rate to "
          << config.integration_rate << std::endl;
        if (config.integration_rate < 1) {
          std::cerr
            << "ERROR: --integration-rate (-r) must >= 1 (was "
            << optarg << ")\n";
          flagErr++;
        }
        break;
      case 's':    //   -s  (--map-size)
        config.volume_size = atof3(optarg);
        std::cerr << "update map_size to " << config.volume_size.x
          << "mx" << config.volume_size.y << "mx"
          << config.volume_size.z << "m" << std::endl;
        if ((config.volume_size.x <= 0) || (config.volume_size.y <= 0)
            || (config.volume_size.z <= 0)) {
          std::cerr
            << "ERROR: --volume-size (-s) all dimensions must > 0 (was "
            << optarg << ")\n";
          flagErr++;
        }
        break;
      case 't':    //   -t  (--tracking-rate)
        config.tracking_rate = atof(optarg);
        std::cerr << "update tracking_rate to " << config.tracking_rate
          << std::endl;
        break;
      case 'z':    //   -z  (--rendering-rate)
        config.rendering_rate = atof(optarg);
        std::cerr << "update rendering_rate to " << config.rendering_rate
          << std::endl;
        break;
      case 'v':    //   -v  (--volumetric-size)
        config.volume_resolution = atoi3(optarg);
        std::cerr << "update volumetric_size to "
          << config.volume_resolution.x << "x"
          << config.volume_resolution.y << "x"
          << config.volume_resolution.z << std::endl;
        if ((config.volume_resolution.x <= 0)
            || (config.volume_resolution.y <= 0)
            || (config.volume_resolution.z <= 0)) {
          std::cerr
            << "ERROR: --volume-size (-s) all dimensions must > 0 (was "
            << optarg << ")\n";
          flagErr++;
        }

        break;
      case 'y': {
                  std::istringstream dotargs(optarg);
                  std::string s;
                  config.pyramid.clear();
                  while (getline(dotargs, s, ',')) {
                    config.pyramid.push_back(atof(s.c_str()));
                  }
                }
                std::cerr << "update pyramid levels to " << pyramid2str(config.pyramid)
                  << std::endl;
                break;
      case 'B':
                config.voxel_block_size = atoi(optarg);
                std::cerr << "update voxel block size to " << config.voxel_block_size << std::endl;
                break;
      case 'F':
                config.bilateralFilter = true;
                std::cerr << "using bilateral filter" << std::endl;
                break;
      case 'C':
                config.colouredVoxels = true;
                std::cerr << "using coloured voxels" << std::endl;
                break;
      case 'M':
                config.multiResolution = true;
                std::cerr << "using multi-resolution integration" << std::endl;
                break;
      case 'S': //   -S  (--mask-rcnn-folder)
              config.maskrcnn_folder = optarg;
              std::cerr << "using mask-rcnn folder: " << config.maskrcnn_folder << std::endl;
              break;
      case 'R': //   -S  (--ground-truth-segment)
        config.gt_mask_folder = optarg;
        std::cerr << "using ground truth folder: " << config.gt_mask_folder <<
                  std::endl;
        break;
      case 'P':
        config.pause = true;
        std::cerr << "pause on each frame, press enter to continue" << std::endl;
        break;
      case 'D':
        config.in_debug = true;
        std::cerr << "in debug mode" << std::endl;
        break;
      case 'O':
        config.output_images = optarg;
        std::cerr << "output rendered images to " <<config.output_images<< std::endl;
        break;
      case 'G':
        config.min_obj_ratio = atof(optarg);
        std::cerr << "allowed minimum object size ratio is " <<config.min_obj_ratio<< std::endl;
        break;
      case 'T':
        config.disable_rgb = true;
        std::cerr << "disbale rgb now: " <<config.disable_rgb<< std::endl;
        break;
      case 'A':
        config.absorb_outlier_bg = true;
        std::cerr << "absorb outliers as background now: " <<config.absorb_outlier_bg<< std::endl;
        break;
      case 'H':
        config.geom_refine_human = true;
        std::cerr << "geom refine human now: " <<config.geom_refine_human << std::endl;
        break;
      case 'V':
        config.obj_moved = atof(optarg);
        std::cerr << "object moved throshold is " <<config.obj_moved<<std::endl;
        break;
      case 'I':
        config.init_frame = atoi(optarg);
        std::cerr << "start process frame from " << config.init_frame << std::endl;
        break;

      case 0:
      case '?':
                std::cerr << "Unknown option character -" << char(optopt)
                  << " or bad usage.\n";
                print_arguments();
                exit(0);
      default:
                std::cerr << "GetOpt abort.";
                flagErr = true;
    }

  if (flagErr) {
    std::cerr << "Exited due to " << flagErr << " error"
      << (flagErr == 1 ? "" : "s")
      << " in command line options\n";
    exit(1);
  }
  return config;
}

#endif /* DEFAULT_PARAMETERS_H_ */
