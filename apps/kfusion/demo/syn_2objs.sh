#!/bin/bash

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT=$HERE/..

cd $ROOT/build
./kfusion-main-openmp --compute-size-ratio 2 --fps 0 --block-read False --input-file /data/midfusion/synthetic/cam0_gt.raw --mask-rcnn-folder /data/midfusion/synthetic/mask_RCNN --icp-threshold 1e-05 --mu 0.075 --init-pose 0.5,0.5,0.4 --integration-rate 1 --volume-size 10 -B 8 --tracking-rate 1 --volume-resolution 512 --pyramid-levels 10,10,10 --rendering-rate 1 -k 600,600,319.5,239.5 -a 0.6574,0.6126,-0.2949,-0.3248 -F --min-obj-ratio 0.01 --absorb-outlier-bg --ground-truth-segment /data/midfusion/synthetic/label0/data/

