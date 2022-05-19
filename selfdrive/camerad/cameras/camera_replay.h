#pragma once

#include "selfdrive/camerad/cameras/camera_common.h"
#include "selfdrive/ui/replay/framereader.h"

#include "cuda_runtime.h"
#include "cuda_runtime_api.h"

#define FRAME_BUF_COUNT 16

#define checkMsg(msg) __checkMsg(msg, __FILE__, __LINE__)
#define checkMsgNoFail(msg) __checkMsgNoFail(msg, __FILE__, __LINE__)

typedef struct CameraState {
  int camera_num;
  CameraInfo ci;

  int fps;
  float digital_gain = 0;

  CameraBuf buf;
  FrameReader *frame = nullptr;
} CameraState;

typedef struct MultiCameraState {
  CameraState road_cam;
  CameraState driver_cam;

  SubMaster *sm = nullptr;
  PubMaster *pm = nullptr;
} MultiCameraState;
