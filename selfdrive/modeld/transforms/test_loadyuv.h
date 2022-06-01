#pragma once

#include "selfdrive/common/mat.h"
#include <inttypes.h>
#include "cuda_runtime.h"
#include "cuda_runtime_api.h"
#include <math.h>

#define checkMsg(msg) __checkMsg(msg, __FILE__, __LINE__)
#define checkMsgNoFail(msg) __checkMsgNoFail(msg, __FILE__, __LINE__)

void test_loadyuv();