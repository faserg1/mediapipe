#include <thread>
#include <sys/types.h>
#include <sys/socket.h>
#include <sys/poll.h>
#include <netdb.h>
#include <endian.h>
#include <chrono>

#include "absl/log/absl_log.h"
#include "l2d_tracker/core/sender.h"

using packet_timestamp_type = uint64_t;
using packet_seq_type = uint64_t;
using packet_size = uint64_t;
using packet_big_endianess = uint8_t;

constexpr const size_t hands_using = 2;

constexpr const size_t header_size = 
    sizeof(packet_big_endianess) + 
    sizeof(packet_size) + 
    sizeof(packet_timestamp_type) + 
    sizeof(packet_seq_type);


constexpr const size_t handedness = sizeof(float) * 2; // <-1 = left, >+1 = right, 0 = uknown
constexpr const size_t hands_size = sizeof(float) * 5 * 21 * hands_using;
constexpr const size_t pose_size = sizeof(float) * 5 * 33;
constexpr const size_t face_size = sizeof(float) * 5 * 478;
constexpr const size_t classification_size = sizeof(float) * 52;
constexpr const size_t classification_offset = header_size;
constexpr const size_t pose_offset = classification_offset + classification_size;
constexpr const size_t hands_offset = pose_offset + pose_size;
constexpr const size_t handedness_offset = hands_offset + hands_size;
constexpr const size_t face_offset = handedness_offset + handedness;
constexpr const size_t total_size = header_size + classification_size + pose_size + hands_size + handedness + face_size;

struct L2DTrackerPack
{
    int socket = 0;
    packet_seq_type seq = 0;
    std::array<std::byte, total_size> packet;

    sockaddr_in toAddr {};
    socklen_t addrLen = 0;
    packet_timestamp_type last_recv_ts = 0;
};

void writeHeader(L2DTrackerPack &packet, packet_timestamp_type now);
void handleIncome(L2DTrackerPack &packet, packet_timestamp_type now);
void writeLandmarks(L2DTrackerPack &packet, const LandmarksType &landmarks, size_t &offset);
packet_timestamp_type get_now();

std::shared_ptr<L2DTrackerPack> createPacket()
{
    auto pack = std::make_shared<L2DTrackerPack>();

    addrinfo hints {}, *result {};
    memset(&hints, 0, sizeof(hints));
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_DGRAM;
    hints.ai_flags = AI_PASSIVE;

    auto addrResult = getaddrinfo(NULL, "54050", &hints, &result);
    if (addrResult != 0)
    {
        ABSL_LOG(ERROR) << "Unable to getaddrinfo: " << gai_strerror(addrResult);
        return {};
    }

    for (auto rp = result; rp != NULL; rp = rp->ai_next) {
        pack->socket = socket(result->ai_family, result->ai_socktype, result->ai_protocol);
        if (pack->socket < 0)
            continue;

        if (bind(pack->socket, rp->ai_addr, rp->ai_addrlen) == 0)
            break;

        close(pack->socket);
    }

    freeaddrinfo(result);
    return pack;
}

void addFaceInfoToPacket(L2DTrackerPack &packet, const mediapipe::ClassificationList &classificationList)
{
    //ABSL_LOG(INFO) << "classification";

    for (int i = 0; i < classificationList.classification_size(); ++i)
    {
        auto &classification = classificationList.classification(i);
        auto *value = reinterpret_cast<float*>(&packet.packet[classification_offset + sizeof(float) * i]);
        *value = classification.score();
    }
}

void addFaceLandmarksToPacket(L2DTrackerPack &packet, const std::vector<LandmarksType> &landmarks)
{
    // Only one fucking face
    auto &first_face = landmarks.front();
    auto offset = face_offset;

    writeLandmarks(packet, first_face, offset);
}

void addHandsInfoToPacket(L2DTrackerPack &packet, const std::vector<LandmarksType> &landmarks)
{
    size_t hands_count = 0;
    size_t offset = hands_offset;
    for (auto &hand : landmarks)
    {
        if (++hands_count > hands_using)
            break;
        writeLandmarks(packet, hand, offset);
    }
}

void addHandednessInfoToPacket(L2DTrackerPack &packet, const std::vector<mediapipe::ClassificationList> &classification)
{
    size_t hands_count = 0;
    for (auto &hand : classification)
    {
        if (++hands_count > hands_using)
            break;
        auto &handedness = hand.classification(0).label();
        auto *writter = reinterpret_cast<float*>(&packet.packet[handedness_offset]);
        *writter = (handedness[0] == 'l' ? -2 : 2);
        writter++;
    }
}

void addPoseInfoToPacket(L2DTrackerPack &packet, const LandmarksType &landmarks)
{
    auto offset = pose_offset;
    writeLandmarks(packet, landmarks, offset);
}

void sendAndFlushPacket(L2DTrackerPack &packet)
{
    packet.seq++;
    auto now = get_now();
    pollfd sockpollin {
        packet.socket,
        POLLIN,
        0
    };
    if (poll(&sockpollin, 1, 0))
    {
        handleIncome(packet, now);
    }
    writeHeader(packet, now);
    if (packet.addrLen && now < (packet.last_recv_ts + 5000))
    {
        sendto(packet.socket, packet.packet.data(), packet.packet.size(), 0, reinterpret_cast<sockaddr*>(&packet.toAddr), packet.addrLen);
        //ABSL_LOG(INFO) << "Send data";
    }
}

void handleIncome(L2DTrackerPack &packet, packet_timestamp_type now)
{
    std::array<std::byte, 4096> buffer;
    sockaddr_in fromAddr;
    socklen_t addrLen = sizeof(fromAddr);
    auto receivedSize = recvfrom(packet.socket, buffer.data(), buffer.size(), 0, reinterpret_cast<sockaddr*>(&fromAddr), &addrLen);
    if (receivedSize < 0)
    {
        // Error handle?
        return;
    }

    //ABSL_LOG(INFO) << "Received income";

    packet.last_recv_ts = now;
    packet.addrLen = addrLen;
    memcpy(&packet.toAddr, &fromAddr, addrLen);
}

packet_timestamp_type get_now()
{
    std::chrono::time_point<std::chrono::system_clock> now = std::chrono::system_clock::now();
    auto duration = now.time_since_epoch();
    return static_cast<packet_timestamp_type>(std::chrono::duration_cast<std::chrono::milliseconds>(duration).count());
}

void writeHeader(L2DTrackerPack &packet, packet_timestamp_type now)
{
    size_t offset = 0;

    auto *big_endianess = reinterpret_cast<packet_big_endianess*>(&packet.packet[offset]);
    offset += sizeof(packet_big_endianess);
    auto *size = reinterpret_cast<packet_size*>(&packet.packet[offset]);
    offset += sizeof(packet_size);
    auto *timestamp_pack = reinterpret_cast<packet_timestamp_type*>(&packet.packet[offset]);
    offset += sizeof(packet_timestamp_type);
    auto *seq_pack = reinterpret_cast<packet_seq_type*>(&packet.packet[offset]);
    offset += sizeof(packet_seq_type);
# if BYTE_ORDER == LITTLE_ENDIAN
    *big_endianess = 0;
# else
    *big_endianess = 255;
# endif
    *size = total_size - header_size;
    *timestamp_pack = now;
    *seq_pack = packet.seq;
}

void writeLandmarks(L2DTrackerPack &packet, const LandmarksType &landmarks, size_t &offset)
{
    int count = landmarks.landmark_size();
    auto *base = reinterpret_cast<float*>(&packet.packet[offset]);
    auto *writter = base;
    for (int idx = 0; idx < count; idx++)
    {
        auto &landmark = landmarks.landmark(idx);
        *writter = landmark.x();
        writter++;

        *writter = landmark.y();
        writter++;

        *writter = landmark.z();
        writter++;

        *writter = landmark.presence();
        writter++;

        *writter = landmark.visibility();
        writter++;
    }

    offset += reinterpret_cast<size_t>(writter) - reinterpret_cast<size_t>(base);
}