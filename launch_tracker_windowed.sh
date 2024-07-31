#!/bin/sh

GLOG_logtostderr=0 bazel-bin/l2d_tracker/tracker/l2d_tracker_cpu --calculator_graph_config_file=l2d_tracker/graphs/total_detector.pbtxt --show_window=true --capture_device=1
