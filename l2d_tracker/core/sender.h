#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/formats/classification.pb.h"
#include <memory>
#include <vector>

struct L2DTrackerPack;

using LandmarksType = mediapipe::NormalizedLandmarkList;
using WroldLandmarksType = mediapipe::LandmarkList;

std::shared_ptr<L2DTrackerPack> createPacket();

void addFaceInfoToPacket(L2DTrackerPack &packet, const mediapipe::ClassificationList &classification);
void addFaceLandmarksToPacket(L2DTrackerPack &packet, const std::vector<LandmarksType> &landmarks);
void addHandednessInfoToPacket(L2DTrackerPack &packet, const std::vector<mediapipe::ClassificationList> &classification);
void addHandsInfoToPacket(L2DTrackerPack &packet, const std::vector<LandmarksType> &landmarks);
void addPoseInfoToPacket(L2DTrackerPack &packet, const LandmarksType &landmarks);

void sendAndFlushPacket(L2DTrackerPack &packet);