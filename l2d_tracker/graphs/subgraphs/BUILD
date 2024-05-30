# Copyright 2019 The MediaPipe Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

load(
    "//mediapipe/framework/tool:mediapipe_graph.bzl",
    "mediapipe_simple_subgraph",
)

load(
    "//mediapipe/framework/tool:mediapipe_files.bzl",
    "mediapipe_files",
)

licenses(["notice"])

package(default_visibility = ["//visibility:public"])

mediapipe_files(srcs = [
    "face_blendshapes.tflite",
])

mediapipe_simple_subgraph(
    name = "pose_landmarks_to_render_data",
    graph = "pose_landmarks_to_render_data.pbtxt",
    register_as = "PoseLandmarksToRenderData",
    deps = [
        "//mediapipe/calculators/core:concatenate_vector_calculator",
        "//mediapipe/calculators/core:split_proto_list_calculator",
        "//mediapipe/calculators/util:landmarks_to_render_data_calculator",
        "//mediapipe/calculators/util:rect_to_render_scale_calculator",
    ],
)


mediapipe_simple_subgraph(
    name = "hand_renderer_cpu",
    graph = "hand_renderer_cpu.pbtxt",
    register_as = "HandRendererSubgraph",
    deps = [
        "//mediapipe/calculators/core:begin_loop_calculator",
        "//mediapipe/calculators/core:end_loop_calculator",
        "//mediapipe/calculators/core:gate_calculator",
        "//mediapipe/calculators/core:split_vector_calculator",
        "//mediapipe/calculators/util:annotation_overlay_calculator",
        "//mediapipe/calculators/util:collection_has_min_size_calculator",
        "//mediapipe/calculators/util:detections_to_render_data_calculator",
        "//mediapipe/calculators/util:labels_to_render_data_calculator",
        "//mediapipe/calculators/util:landmarks_to_render_data_calculator",
        "//mediapipe/calculators/util:rect_to_render_data_calculator",
    ],
)

mediapipe_simple_subgraph(
    name = "pose_renderer_cpu",
    graph = "pose_renderer_cpu.pbtxt",
    register_as = "PoseRendererCpu",
    deps = [
        ":pose_landmarks_to_render_data",
        "//mediapipe/calculators/image:image_properties_calculator",
        "//mediapipe/calculators/image:recolor_calculator",
        "//mediapipe/calculators/util:annotation_overlay_calculator",
        "//mediapipe/calculators/util:detections_to_render_data_calculator",
        "//mediapipe/calculators/util:rect_to_render_data_calculator",
    ],
)

mediapipe_simple_subgraph(
    name = "face_categorizer",
    graph = "face_categorizer.pbtxt",
    register_as = "FaceCategorizer",
    deps = [
        "//mediapipe/calculators/core:split_vector_calculator",
        "//mediapipe/calculators/core:get_vector_item_calculator",
        "//mediapipe/calculators/tensor:landmarks_to_tensor_calculator",
        "//mediapipe/calculators/tensor:inference_calculator_interface",
        "//mediapipe/calculators/tensor:tensors_to_classification_calculator",
        "//mediapipe/tasks/cc/core:model_resources_calculator",
    ],
)
