#include <cstdlib>
#include <mutex>
#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/log/absl_log.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/formats/classification.pb.h"
#include "mediapipe/framework/port/file_helpers.h"
#include "mediapipe/framework/port/opencv_highgui_inc.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"
#include "mediapipe/framework/port/opencv_video_inc.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/util/resource_util.h"

#include "l2d_tracker/core/sender.h"

constexpr char kInputStream[] = "input_video";
constexpr char kEmptyStream[] = "input_empty";
constexpr char kOutputStream[] = "output_video";
/*
constexpr char kHandsLandmarks[] = "hand_world_landmarks";
constexpr char kPoseLandmarks[] = "pose_world_landmarks";
*/
constexpr char kHandsLandmarks[] = "hand_landmarks";
constexpr char kPoseLandmarks[] = "pose_landmarks";

constexpr char kHandedness[] = "handedness";
constexpr char kFaceBlendshapes[] = "blendshapes";

constexpr char kWindowName[] = "L2D MediaPipe Tracker Test";

ABSL_FLAG(std::string, calculator_graph_config_file, "",
          "Name of file containing text format CalculatorGraphConfig proto.");

ABSL_FLAG(bool, show_window, false, 
    "true, if need to show window with landmarks.");

namespace {
  bool runGraph = true;
}

std::shared_ptr<L2DTrackerPack> packet;

absl::Status handleHandedness(const mediapipe::Packet &packetHandedness)
{
  if (!packet)
    return {};
  if (packetHandedness.IsEmpty())
    return {};
  auto &handednessList = packetHandedness.Get<std::vector<mediapipe::ClassificationList>>();
  addHandednessInfoToPacket(*packet, handednessList);
  return {};
}

absl::Status handleHands(const mediapipe::Packet &packetHands)
{
  if (!packet)
    return {};
  if (packetHands.IsEmpty())
    return {};
  auto &handsList = packetHands.Get<std::vector<LandmarksType>>();
  addHandsInfoToPacket(*packet, handsList);
  return {};
}

absl::Status handlePose(const mediapipe::Packet &packetPose)
{
  if (!packet)
    return {};
  if (packetPose.IsEmpty())
    return {};
  auto &poseLandmarks = packetPose.Get<LandmarksType>();
  addPoseInfoToPacket(*packet, poseLandmarks);
  return {};
}

absl::Status handleBlendshapes(const mediapipe::Packet &packetBlendshapes)
{
  if (packetBlendshapes.IsEmpty())
    return {};

  if (!packet)
    return {};

  auto &classificationList = packetBlendshapes.Get<mediapipe::ClassificationList>();
  addFaceInfoToPacket(*packet, classificationList);

  return {};
}

void handleError(const absl::Status &status)
{
  if (!status.ok()) {
    ABSL_LOG(ERROR) << "Failed to run the graph: \"" << status.message() << "\", code: " << status.code();
  }
}

absl::Status RunMPPGraph() {
  std::string calculator_graph_config_contents;
  MP_RETURN_IF_ERROR(mediapipe::file::GetContents(
      absl::GetFlag(FLAGS_calculator_graph_config_file),
      &calculator_graph_config_contents));
  /*ABSL_LOG(INFO) << "Get calculator graph config contents: "
                 << calculator_graph_config_contents;*/
  mediapipe::CalculatorGraphConfig config =
      mediapipe::ParseTextProtoOrDie<mediapipe::CalculatorGraphConfig>(
          calculator_graph_config_contents);

  ABSL_LOG(INFO) << "Initialize the calculator graph.";
  mediapipe::CalculatorGraph graph;
  MP_RETURN_IF_ERROR(graph.Initialize(config));
  

  ABSL_LOG(INFO) << "Initialize the camera.";

  cv::VideoCapture capture;
  capture.open(0, cv::CAP_V4L);
  RET_CHECK(capture.isOpened());

  const bool show_window = absl::GetFlag(FLAGS_show_window);
  if (show_window) {
    cv::namedWindow(kWindowName, /*flags=WINDOW_AUTOSIZE*/ 1);
  }
#if (CV_MAJOR_VERSION >= 3) && (CV_MINOR_VERSION >= 2)
  int fourccSet = mediapipe::fourcc('M','J','P','G');
  //capture.set(cv::CAP_PROP_FOURCC, fourccSet);
  capture.set(cv::CAP_PROP_FRAME_WIDTH, 864);
  capture.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
  
  
  capture.set(cv::CAP_PROP_FPS, 20);
  //
  

  auto w = capture.get(cv::CAP_PROP_FRAME_WIDTH);
  auto h = capture.get(cv::CAP_PROP_FRAME_HEIGHT);
  auto fps = capture.get(cv::CAP_PROP_FPS);
  auto fourcc = capture.get(cv::CAP_PROP_FOURCC);

  ABSL_LOG(INFO) << "Camera settings: " << w << "x" << h << "*" << fps << ", fourcc: " << fourcc << ".";
#endif

  ABSL_LOG(INFO) << "Checking if camera alive after applied settings.";
  RET_CHECK(capture.isOpened());

  ABSL_LOG(INFO) << "Start running the calculator graph.";

  if (!show_window) {
    graph.SetGraphInputStreamAddMode(mediapipe::CalculatorGraph::GraphInputStreamAddMode::ADD_IF_NOT_FULL);

    graph.SetInputStreamMaxQueueSize(kInputStream, 3);
    graph.SetInputStreamMaxQueueSize(kEmptyStream, 3);
  }

  // TODO: ObserveOutputStream

  std::shared_ptr<mediapipe::OutputStreamPoller> pollerFrame;

  if (show_window) {
    auto status_poller = graph.AddOutputStreamPoller(kOutputStream);
    if (!status_poller.ok()) {
      return {};
    }

    pollerFrame = std::make_shared<mediapipe::OutputStreamPoller>(std::move(status_poller.value()));
  }

  MP_RETURN_IF_ERROR(graph.ObserveOutputStream(kPoseLandmarks, &handlePose, true));
  MP_RETURN_IF_ERROR(graph.ObserveOutputStream(kHandsLandmarks, &handleHands, true));
  MP_RETURN_IF_ERROR(graph.ObserveOutputStream(kFaceBlendshapes, &handleBlendshapes, true));

  graph.SetErrorCallback(&handleError);

  ABSL_LOG(INFO) << "Creating network connection.";

  packet = createPacket();

  MP_RETURN_IF_ERROR(graph.StartRun({}));

  ABSL_LOG(INFO) << "Start grabbing and processing frames.";
  while (runGraph) {
    //ABSL_LOG(INFO) << "0";
    // Capture opencv camera or video frame.
    cv::Mat camera_frame_raw;
    capture >> camera_frame_raw;
    if (camera_frame_raw.empty()) {
      ABSL_LOG(INFO) << "Ignore empty frames from camera.";
      continue;
    }
    //ABSL_LOG(INFO) << "1";
    cv::Mat camera_frame;
    cv::cvtColor(camera_frame_raw, camera_frame, cv::COLOR_BGR2RGB);
    //cv::flip(camera_frame, camera_frame, /*flipcode=HORIZONTAL*/ 1);

    // Wrap Mat into an ImageFrame.
    auto input_frame = absl::make_unique<mediapipe::ImageFrame>(
        mediapipe::ImageFormat::SRGB, camera_frame.cols, camera_frame.rows,
        mediapipe::ImageFrame::kDefaultAlignmentBoundary);
    cv::Mat input_frame_mat = mediapipe::formats::MatView(input_frame.get());
    camera_frame.copyTo(input_frame_mat);

    //ABSL_LOG(INFO) << "2";

    

    // Send image packet into the graph.
    size_t frame_timestamp_us =
        (double)cv::getTickCount() / (double)cv::getTickFrequency() * 1e6;

    //ABSL_LOG(INFO) << "3";
      
    MP_RETURN_IF_ERROR(graph.AddPacketToInputStream(
        kInputStream, mediapipe::Adopt(input_frame.release())
                          .At(mediapipe::Timestamp(frame_timestamp_us))));

    if (show_window) {
      // Create an empty image
      auto empty_frame = absl::make_unique<mediapipe::ImageFrame>(
          mediapipe::ImageFormat::SRGB, camera_frame.cols, camera_frame.rows,
          mediapipe::ImageFrame::kDefaultAlignmentBoundary);

      MP_RETURN_IF_ERROR(graph.AddPacketToInputStream(
          kEmptyStream, mediapipe::Adopt(empty_frame.release())
                            .At(mediapipe::Timestamp(frame_timestamp_us))));
    }

    

    //ABSL_LOG(INFO) << "4";

    //graph.WaitForObservedOutput();

    //ABSL_LOG(INFO) << "Running...";

    // Get the graph result packet, or stop if that fails.
  
    if (show_window && pollerFrame) {
      mediapipe::Packet packetFrame;

      if (!pollerFrame->Next(&packetFrame))
        break;

      auto& output_frame = packetFrame.Get<mediapipe::ImageFrame>();

      // Convert back to opencv for display or saving.
      cv::Mat output_frame_mat = mediapipe::formats::MatView(&output_frame);
      cv::cvtColor(output_frame_mat, output_frame_mat, cv::COLOR_RGB2BGR);

      cv::imshow(kWindowName, output_frame_mat);

      // Press any key to exit.
      const int pressed_key = cv::waitKey(5);
      if (pressed_key >= 0 && pressed_key != 255)
        runGraph = false;
        //ABSL_LOG(INFO) << "Closing...";
    }

    //ABSL_LOG(INFO) << "7";

    if (packet)
      sendAndFlushPacket(*packet);

    //ABSL_LOG(INFO) << "8";
  }

  ABSL_LOG(INFO) << "Shutting down.";
  MP_RETURN_IF_ERROR(graph.CloseInputStream(kInputStream));
  MP_RETURN_IF_ERROR(graph.CloseInputStream(kEmptyStream));
  packet.reset();
  return graph.WaitUntilDone();
}

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  absl::ParseCommandLine(argc, argv);
  absl::Status run_status = RunMPPGraph();
  if (!run_status.ok()) {
    ABSL_LOG(ERROR) << "Failed to run the graph: \"" << run_status.message() << "\", code: " << run_status.code();
    return EXIT_FAILURE;
  } else {
    ABSL_LOG(INFO) << "Success!";
  }
  return EXIT_SUCCESS;
}
