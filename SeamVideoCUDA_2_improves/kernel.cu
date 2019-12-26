#include "kernel.h"

__global__ void seam_gpu_kernel(const PtrStepSz<uchar3> src1, const PtrStepSz<uchar3> src2, PtrStep<uchar3> dst, const int* seamline)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	//int offset = x + y * blockDim.x * gridDim.x;

	if (x < src1.cols && y < src1.rows)
	{
		//uchar3 v = src1(y, x);
		uchar3 v;
		if (x < seamline[y])
			v = src1(y, x);
		else
			v = src2(y, x);
		dst(y, x) = make_uchar3(v.x, v.y, v.z);
	}
}


void seam_gpu_caller(const PtrStepSz<uchar3>& src1, const PtrStepSz<uchar3>& src2, PtrStep<uchar3> dst, const int* seamline, cudaStream_t stream)
{
	dim3 block(32, 8);
	dim3 grid((src1.cols + block.x - 1) / block.x, (src1.rows + block.y - 1) / block.y);

	seam_gpu_kernel << <grid, block, 0, stream >> >(src1, src2, dst, seamline);
	if (stream == 0)
		cudaDeviceSynchronize();
}


void seam_gpu(const GpuMat& src1, const GpuMat& src2, GpuMat& dst, const vector<int>& gseamLine, Stream& stream)
{
	dst.create(src1.size(), src1.type());
	cudaStream_t s = StreamAccessor::getStream(stream);

	int sz = gseamLine.size();
	int* seam = new int[sz];

//#pragma omp parallel for
	for (int i = 0; i < sz; i++)
		seam[i] = gseamLine[i];
	int* seamline;
	cudaMalloc((void**)&seamline, sz*sizeof(int));
	cudaMemcpy(seamline, seam, sz*sizeof(int), cudaMemcpyHostToDevice);

	seam_gpu_caller(src1, src2, dst, seamline, s);

	cudaFree(seamline);
	free(seam);
}