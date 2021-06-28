#!/usr/bin/env bash

git pull

~/experiment_start node-1 yolo train_and_eval yolo_custom yolov4/debug/512-baseline i$1 &
sleep 10m

~/experiment_start node-2 yolo train_and_eval yolo_custom yolov4/debug/512-btest i$1 &
sleep 10m

~/experiment_start node-3 yolo train_and_eval yolo_custom yolov4/debug/512-jitter i$1 &
sleep 10m

~/experiment_start node-4 yolo train_and_eval yolo_custom yolov4/debug/512-jitter-resize i$1 &
sleep 10m

~/experiment_start node-5 yolo train_and_eval yolo_custom yolov4/debug/512-jitter-scale i$1 &
sleep 10m

~/experiment_start node-6 yolo train_and_eval yolo_custom yolov4/debug/512-jitter-scratch i$1 &
sleep 10m

~/experiment_start node-7 yolo train_and_eval yolo_custom yolov4/debug/512-jitter-crop-scale i$1 &
sleep 10m