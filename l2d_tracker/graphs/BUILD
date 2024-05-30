load(
    "//mediapipe/framework/tool:mediapipe_graph.bzl",
    "mediapipe_binary_graph",
)

licenses(["notice"])

package(default_visibility = ["//visibility:public"])

exports_files(glob([
    "*.pbtxt",
]))

cc_library(
    name = "all_calculators",
    data = [
        "//l2d_tracker/graphs/subgraphs:face_blendshapes.tflite",
        "//mediapipe/modules/face_landmark:face_landmark.tflite",
    ],
    deps = [
        "//mediapipe/calculators/core:constant_side_packet_calculator",
        "//mediapipe/calculators/core:merge_calculator",
        "//mediapipe/calculators/core:flow_limiter_calculator",
        "//mediapipe/calculators/core:gate_calculator",
        "//mediapipe/calculators/core:immediate_mux_calculator",
        "//mediapipe/calculators/core:packet_inner_join_calculator",
        "//mediapipe/calculators/core:previous_loopback_calculator",

        "//mediapipe/calculators/video:opencv_video_decoder_calculator",
        "//mediapipe/calculators/video:opencv_video_encoder_calculator",

        #"//mediapipe/graphs/hand_tracking/subgraphs:hand_renderer_cpu",
        "//l2d_tracker/graphs/subgraphs:hand_renderer_cpu",
        "//l2d_tracker/graphs/subgraphs:pose_renderer_cpu",
        "//l2d_tracker/graphs/subgraphs:face_categorizer",
        
        
        
        
        "//mediapipe/graphs/face_mesh/subgraphs:face_renderer_cpu",
        
        "//mediapipe/modules/hand_landmark:hand_landmark_tracking_cpu",
        "//mediapipe/modules/face_landmark:face_landmark_front_cpu",
        "//mediapipe/modules/pose_landmark:pose_landmark_cpu",
    ],
)

mediapipe_binary_graph(
    name = "total_detector_binary_graph",
    graph = "total_detector.pbtxt",
    output_name = "total_detector.binarypb",
    deps = [":all_calculators"],
)

mediapipe_binary_graph(
    name = "total_detector_invisible_binary_graph",
    graph = "total_detector_invisible.pbtxt",
    output_name = "total_detector_invisible.binarypb",
    deps = [":all_calculators"],
)
