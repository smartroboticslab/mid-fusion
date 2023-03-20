/*
 * SPDX-FileCopyrightText: 2016-2019 Emanuele Vespa
 * SPDX-FileCopyrightText: 2017-2019 Smart Robotics Lab, Imperial College London
 * SPDX-FileCopyrightText: 2017-2019 Binbin Xu
 * SPDX-License-Identifier: BSD-3-Clause
*/

#ifndef CONFIG_H
#define CONFIG_H

#include <vector_types.h>
#include <vector>
#include <string>

struct Configuration {

	// 
  // KFusion configuration parameters
  // Command line arguments are parsed in default_parameters.h 
  //

	int compute_size_ratio;
	int integration_rate;
	int rendering_rate;
	int tracking_rate;
	uint3 volume_resolution;
	float3 volume_size;
    int voxel_block_size;
	float3 initial_pos_factor;
	std::vector<int> pyramid;
	std::string dump_volume_file;
	std::string input_file;
	std::string log_file;
  std::string groundtruth_file;

	float4 camera;
	bool camera_overrided;

	float mu;
	int fps;
	bool blocking_read;
	float icp_threshold;
	bool no_gui;
	bool render_volume_fullsize;
  bool bilateralFilter;
  bool colouredVoxels;
  bool multiResolution;
  bool bayesian;

	//for inputing mask-rcnn segmentation
	std::string maskrcnn_folder;
	std::string gt_mask_folder;

	bool pause;
	bool in_debug;
	std::string output_images;
	float min_obj_ratio;
	bool disable_rgb;
	bool absorb_outlier_bg;
	bool geom_refine_human;
  float obj_moved;
  int init_frame;
};

#endif
