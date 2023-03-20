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
#include <vector>
#include <sstream>
#include <string>
#include <cstring>
#include <time.h>
#include <csignal>

#include <sys/types.h>
#include <sys/stat.h>
#include <sstream>
#include <iomanip>
#include <getopt.h>
#include <perfstats.h>

PerfStats Stats;



/***
 * This program loop over a scene recording
 */

int main(int argc, char ** argv) {

	Configuration config = parseArgs(argc, argv);

	// ========= CHECK ARGS =====================

	std::ostream* logstream = &std::cout;
	std::ofstream logfilestream;
	assert(config.compute_size_ratio > 0);
	assert(config.integration_rate > 0);
	assert(config.volume_size.x > 0);
	assert(config.volume_resolution.x > 0);

	if (config.log_file != "") {
		logfilestream.open(config.log_file.c_str());
		logstream = &logfilestream;
	}
	if (config.input_file == "") {
		std::cerr << "No input found." << std::endl;
		print_arguments();
		exit(1);
	}

	// ========= READER INITIALIZATION  =========

	DepthReader * reader;

	if (is_file(config.input_file)) {
		reader = new RawDepthReader(config.input_file, config.fps,
				config.blocking_read);

	} else {
		reader = new SceneDepthReader(config.input_file, config.fps,
				config.blocking_read);
	}

	std::cout.precision(10);
	std::cerr.precision(10);

	float3 init_pose = config.initial_pos_factor * config.volume_size;
	const uint2 inputSize = reader->getinputSize();
	std::cerr << "input Size is = " << inputSize.x << "," << inputSize.y
			<< std::endl;

	//  =========  BASIC PARAMETERS  (input size / computation size )  =========

	const uint2 computationSize = make_uint2(
			inputSize.x / config.compute_size_ratio,
			inputSize.y / config.compute_size_ratio);
	float4 camera = reader->getK() / config.compute_size_ratio;

	if (config.camera_overrided)
		camera = config.camera / config.compute_size_ratio;
	//  =========  BASIC BUFFERS  (input / output )  =========

	// Construction Scene reader and input buffer
	uint16_t* inputDepth = (uint16_t*) malloc(
			sizeof(uint16_t) * inputSize.x * inputSize.y);
	uchar4* depthRender = (uchar4*) malloc(
			sizeof(uchar4) * computationSize.x * computationSize.y);
	uchar4* trackRender = (uchar4*) malloc(
			sizeof(uchar4) * computationSize.x * computationSize.y);
	uchar4* volumeRender = (uchar4*) malloc(
			sizeof(uchar4) * computationSize.x * computationSize.y);

	uint frame = 0;

	Kfusion kfusion(computationSize, config.volume_resolution,
			config.volume_size, init_pose, config.pyramid, config);
     
  bool bilateralfilter = false;
	double timings[7];
	timings[0] = tock();

	*logstream
			<< "frame\tacquisition\tpreprocessing\ttracking\tintegration\traycasting\trendering\tcomputation\ttotal    \tX          \tY          \tZ         \ttracked   \tintegrated"
			<< std::endl;
	logstream->setf(std::ios::fixed, std::ios::floatfield);

    while (reader->readNextDepthFrame(inputDepth)) {

		timings[1] = tock();

		kfusion.preprocessing(inputDepth, inputSize, config.bilateralFilter);

		timings[2] = tock();

		bool tracked = kfusion.tracking(camera, config.tracking_rate, frame);


		timings[3] = tock();

		Matrix4 pose = kfusion.getPose();

		float xt = pose.data[0].w - init_pose.x;
		float yt = pose.data[1].w - init_pose.y;
		float zt = pose.data[2].w - init_pose.z;


		bool integrated = kfusion.integration(camera, config.integration_rate,
				config.mu, frame);

		timings[4] = tock();

		bool raycast = kfusion.raycasting(camera, config.mu, frame);

		timings[5] = tock();

		kfusion.renderDepth(depthRender, computationSize);
		kfusion.renderTrack(trackRender, computationSize);
		kfusion.renderVolume(volumeRender, computationSize, frame,
				config.rendering_rate, camera, 0.75 * config.mu, true);

		timings[6] = tock();

		*logstream << frame << "\t" << timings[1] - timings[0] << "\t" //  acquisition
				<< timings[2] - timings[1] << "\t"     //  preprocessing
				<< timings[3] - timings[2] << "\t"     //  tracking
				<< timings[4] - timings[3] << "\t"     //  integration
				<< timings[5] - timings[4] << "\t"     //  raycasting
				<< timings[6] - timings[5] << "\t"     //  rendering
				<< timings[5] - timings[1] << "\t"     //  computation
				<< timings[6] - timings[0] << "\t"     //  total
				<< xt << "\t" << yt << "\t" << zt << "\t"     //  X,Y,Z
				<< tracked << "        \t" << integrated // tracked and integrated flags
				<< std::endl;

		frame++;

		timings[0] = tock();
	}
    
    // ==========     DUMP VOLUME      =========

  if (config.dump_volume_file != "") {
    double s = tock();
    kfusion.dumpVolume(config.dump_volume_file);
    double e = tock();
    std::cout << "Mesh generated in " << e - s << " seconds" << std::endl;
  }

  kfusion.printStats();
  
  
    // kfusion.getPointCloudFromVolume(config.mu);

	//  =========  FREE BASIC BUFFERS  =========

	free(inputDepth);
	free(depthRender);
	free(trackRender);
	free(volumeRender);
	return 0;

}

