#!/bin/sh

bazel build -c opt --define MEDIAPIPE_DISABLE_GPU=1 l2d_tracker/tracker:l2d_tracker_cpu