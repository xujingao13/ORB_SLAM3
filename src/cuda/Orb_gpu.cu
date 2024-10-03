/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#include "opencv2/core/cuda/common.hpp"
#include "opencv2/core/cuda/utility.hpp"
#include "opencv2/core/cuda/reduce.hpp"
#include "opencv2/core/cuda/functional.hpp"
#include <cuda/helper_cuda.h>
#include <cuda/Orb.hpp>
#include <Utils.hpp>
#include <cuda_runtime.h>
#include <opencv2/core/cuda.hpp>
#include <opencv2/core/cuda_stream_accessor.hpp>

using namespace cv;
using namespace cv::cuda;
using namespace cv::cuda::device;

namespace ORB_SLAM3 { namespace cuda {

  __constant__ unsigned char c_pattern[sizeof(Point) * 512];

  void GpuOrb::loadPattern(const Point * _pattern) {
    checkCudaErrors( cudaMemcpyToSymbol(c_pattern, _pattern, sizeof(Point) * 512) );
  }

#define GET_VALUE(idx) \
    image(loc.y + __float2int_rn(pattern[idx].x * b + pattern[idx].y * a), \
          loc.x + __float2int_rn(pattern[idx].x * a - pattern[idx].y * b))

  __global__ void calcOrb_kernel(const PtrStepb image, KeyPoint * keypoints, const int npoints, PtrStepb descriptors) {
    int id = blockIdx.x;
    int tid = threadIdx.x;
    if (id >= npoints) return;

    const KeyPoint &kpt = keypoints[id];
    short2 loc = make_short2(kpt.pt.x, kpt.pt.y);
    const Point * pattern = ((Point *)c_pattern) + 16 * tid;

    uchar * desc = descriptors.ptr(id);
    const float factorPI = (float)(CV_PI/180.f);
    float angle = (float)kpt.angle * factorPI;
    float a = (float)cosf(angle), b = (float)sinf(angle);

    int t0, t1, val;
    t0 = GET_VALUE(0); t1 = GET_VALUE(1);
    val = t0 < t1;
    t0 = GET_VALUE(2); t1 = GET_VALUE(3);
    val |= (t0 < t1) << 1;
    t0 = GET_VALUE(4); t1 = GET_VALUE(5);
    val |= (t0 < t1) << 2;
    t0 = GET_VALUE(6); t1 = GET_VALUE(7);
    val |= (t0 < t1) << 3;
    t0 = GET_VALUE(8); t1 = GET_VALUE(9);
    val |= (t0 < t1) << 4;
    t0 = GET_VALUE(10); t1 = GET_VALUE(11);
    val |= (t0 < t1) << 5;
    t0 = GET_VALUE(12); t1 = GET_VALUE(13);
    val |= (t0 < t1) << 6;
    t0 = GET_VALUE(14); t1 = GET_VALUE(15);
    val |= (t0 < t1) << 7;

    // if (id == 0 && tid == 0)
    // {
    //   printf("GPU - val: %x\n", val);
    // }
    desc[tid] = (uchar)val;
  }

#undef GET_VALUE

  GpuOrb::GpuOrb(int maxKeypoints) : maxKeypoints(maxKeypoints), descriptors(maxKeypoints, 32, CV_8UC1) {
    checkCudaErrors( cudaStreamCreate(&stream) );
    cvStream = StreamAccessor::wrapStream(stream);
    checkCudaErrors( cudaMalloc(&keypoints, sizeof(KeyPoint) * maxKeypoints) );
  }

  GpuOrb::~GpuOrb() {
    cvStream.~Stream();
    checkCudaErrors( cudaFree(keypoints) );
    checkCudaErrors( cudaStreamDestroy(stream) );
  }

  void GpuOrb::launch_async(InputArray _image, const KeyPoint * _keypoints, const int npoints) {
    if (npoints == 0) {
      POP_RANGE;
      return ;
    }
    const GpuMat image = _image.getGpuMat();

    checkCudaErrors( cudaMemcpyAsync(keypoints, _keypoints, sizeof(KeyPoint) * npoints, cudaMemcpyHostToDevice, stream) );
    desc = descriptors.rowRange(0, npoints);
    desc.setTo(Scalar::all(0), cvStream);

    dim3 dimBlock(32);
    dim3 dimGrid(npoints);
    calcOrb_kernel<<<dimGrid, dimBlock, 0, stream>>>(image, keypoints, npoints, desc);
    checkCudaErrors( cudaGetLastError() );
  }

  void GpuOrb::join(Mat & _descriptors) {
    desc.download(_descriptors, cvStream);
    checkCudaErrors( cudaStreamSynchronize(stream) );
  }
} }

// namespace ORB_SLAM3 { namespace cuda {

// __constant__ unsigned char c_pattern[sizeof(Point) * 512];

// void GpuOrb::loadPattern(const Point * _pattern) {
//     checkCudaErrors( cudaMemcpyToSymbol(c_pattern, _pattern, sizeof(Point) * 512) );
// }

// __device__ inline int get_pixel(const cv::cuda::PtrStepb& image, const Point* pattern, int idx, float a, float b, short2 center)
// {
//     float x = pattern[idx].x * a - pattern[idx].y * b;
//     float y = pattern[idx].x * b + pattern[idx].y * a;
//     return image(__float2int_rn(center.y + y), __float2int_rn(center.x + x));
// }

// __global__ void calcOrb_kernel(const cv::cuda::PtrStepb image, cv::KeyPoint* keypoints, const int npoints, cv::cuda::PtrStepb descriptors) {
//     int id = blockIdx.x;
//     if (id >= npoints) return;

//     const cv::KeyPoint &kpt = keypoints[id];
//     short2 center = make_short2(__float2int_rn(kpt.pt.x), __float2int_rn(kpt.pt.y));

//     uchar* desc = descriptors.ptr(id);
//     const float factorPI = (float)(CV_PI/180.f);
//     float angle = (float)kpt.angle * factorPI;
//     float a = __cosf(angle), b = __sinf(angle);

//     const Point* pattern = (const Point*)c_pattern;

//     for (int i = 0; i < 32; ++i) {
//         int t0, t1, val;
//         t0 = get_pixel(image, pattern, i*16+0, a, b, center);
//         t1 = get_pixel(image, pattern, i*16+1, a, b, center);
//         val = t0 < t1;
//         t0 = get_pixel(image, pattern, i*16+2, a, b, center);
//         t1 = get_pixel(image, pattern, i*16+3, a, b, center);
//         val |= (t0 < t1) << 1;
//         t0 = get_pixel(image, pattern, i*16+4, a, b, center);
//         t1 = get_pixel(image, pattern, i*16+5, a, b, center);
//         val |= (t0 < t1) << 2;
//         t0 = get_pixel(image, pattern, i*16+6, a, b, center);
//         t1 = get_pixel(image, pattern, i*16+7, a, b, center);
//         val |= (t0 < t1) << 3;
//         t0 = get_pixel(image, pattern, i*16+8, a, b, center);
//         t1 = get_pixel(image, pattern, i*16+9, a, b, center);
//         val |= (t0 < t1) << 4;
//         t0 = get_pixel(image, pattern, i*16+10, a, b, center);
//         t1 = get_pixel(image, pattern, i*16+11, a, b, center);
//         val |= (t0 < t1) << 5;
//         t0 = get_pixel(image, pattern, i*16+12, a, b, center);
//         t1 = get_pixel(image, pattern, i*16+13, a, b, center);
//         val |= (t0 < t1) << 6;
//         t0 = get_pixel(image, pattern, i*16+14, a, b, center);
//         t1 = get_pixel(image, pattern, i*16+15, a, b, center);
//         val |= (t0 < t1) << 7;

//         desc[i] = (uchar)val;
//     }
// }

// GpuOrb::GpuOrb(int maxKeypoints) : maxKeypoints(maxKeypoints), descriptors(maxKeypoints, 32, CV_8UC1) {
//   checkCudaErrors( cudaStreamCreate(&stream) );
//   cvStream = StreamAccessor::wrapStream(stream);
//   checkCudaErrors( cudaMalloc(&keypoints, sizeof(KeyPoint) * maxKeypoints) );
// }

// GpuOrb::~GpuOrb() {
//   cvStream.~Stream();
//   checkCudaErrors( cudaFree(keypoints) );
//   checkCudaErrors( cudaStreamDestroy(stream) );
// }

// void GpuOrb::launch_async(cv::InputArray _image, const cv::KeyPoint * _keypoints, const int npoints) {
//     if (npoints == 0) {
//         return;
//     }
//     const cv::cuda::GpuMat image = _image.getGpuMat();

//     checkCudaErrors( cudaMemcpyAsync(keypoints, _keypoints, sizeof(cv::KeyPoint) * npoints, cudaMemcpyHostToDevice, stream) );
//     desc = descriptors.rowRange(0, npoints);
//     desc.setTo(cv::Scalar::all(0), cvStream);

//     dim3 block(1);
//     dim3 grid(npoints);
//     calcOrb_kernel<<<grid, block, 0, stream>>>(image, keypoints, npoints, desc);
//     checkCudaErrors( cudaGetLastError() );
// }

// void GpuOrb::join(cv::Mat &_descriptors) {
//     if (desc.empty()) return;
//     desc.download(_descriptors, cvStream);
//     cvStream.waitForCompletion();
// }

// } }  // namespace ORB_SLAM3::cuda