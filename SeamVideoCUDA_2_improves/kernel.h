#pragma once
#include <opencv2\opencv.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\core\core.hpp>
#include <opencv2\nonfree\nonfree.hpp>
#include <opencv2\legacy\legacy.hpp>
#include <stdio.h>
#include <ctime>
#include <omp.h>

#include <opencv2\gpu\gpu.hpp>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <opencv2/gpu/stream_accessor.hpp>
#include "GetHomography.h"
#include "StitchAndSeam.h"

using namespace std;
using namespace cv;
using namespace cv::gpu;

//__global__ void seam_gpu_kernel(const PtrStepSz<uchar3> src1, const PtrStepSz<uchar3> src2, PtrStep<uchar3> dst, const int* seamline);

void seam_gpu_caller(const PtrStepSz<uchar3>& src1, const PtrStepSz<uchar3>& src2, PtrStep<uchar3> dst, const int* seamline, cudaStream_t stream);
void seam_gpu(const GpuMat& src1, const GpuMat& src2, GpuMat& dst, const vector<int>& gseamLine, Stream& stream = Stream::Null());