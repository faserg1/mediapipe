licenses(["notice"])

package(default_visibility = [
    "//visibility:public",
])


cc_binary(
    name = "l2d_tracker_files",
    data = [
        "//mediapipe/modules/hand_landmark:hand_landmark_full.tflite",
        "//mediapipe/modules/palm_detection:palm_detection_full.tflite",
        "//mediapipe/modules/face_landmark:face_landmark_with_attention.tflite",
        "//mediapipe/modules/pose_detection:pose_detection.tflite",
        "//mediapipe/modules/pose_landmark:pose_landmark_full.tflite",
    ],
    deps = [
        #"//mediapipe/examples/desktop:simple_run_graph_main",
        #"//mediapipe/graphs/hand_tracking:desktop_tflite_calculators",
        "//l2d_tracker/graphs:all_calculators",
    ],
)

cc_binary(
    name = "l2d_tracker_cpu",
    data = [
        "//mediapipe/modules/hand_landmark:hand_landmark_full.tflite",
        "//mediapipe/modules/palm_detection:palm_detection_full.tflite",
        "//mediapipe/modules/face_landmark:face_landmark_with_attention.tflite",
        "//mediapipe/modules/pose_detection:pose_detection.tflite",
        "//mediapipe/modules/pose_landmark:pose_landmark_full.tflite",
    ],
    deps = [
        "//l2d_tracker/core:l2d_tracker_core",
        #"//mediapipe/graphs/hand_tracking:desktop_tflite_calculators",
        "//l2d_tracker/graphs:all_calculators",
    ],
)