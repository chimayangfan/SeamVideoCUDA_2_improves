#ifndef STITCHANDSEAM_H
#define STITCHANDSEAM_H



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

using namespace std;
using namespace cv;
using namespace cv::gpu;

//aviתyuv
void WriteYuv(string& str1, char* save);
//��Ƶƴ��
void VideoStitch(char* str1, char* str2, int width, int height);
//��ƵͼƬ��һ��ƴ��
Mat imageStitchForVideo(Mat& frame1, Mat& frame2);
//��ƵͼƬ����ƴ��
Mat imageStitchForVideoLater(Mat& frame1, Mat& frame2);
//��ƵͼƬ���������
GpuMat seamCuttingForVideo(GpuMat& leftOverlap_gpu, GpuMat& rightOverlap_gpu);
//������ں�
Mat seamCutting(Mat& leftOverlap, Mat& rightOverlap);

#endif