/* This software contains source code provided by NVIDIA Corporation.
 * You can find the copyright notice below*/
/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */


// Includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#ifdef _WIN32
#  define WINDOWS_LEAN_AND_MEAN
#  define NOMINMAX
#  include <windows.h>
#endif

#include <opencv2/imgproc/imgproc_c.h>

#include "get_cells.h"
#include "get_features.h"

// Includes CUDA
#include <cuda_runtime.h>

// Utilities and timing functions
#include <helper_functions.h>    // includes cuda.h and cuda_runtime_api.h

// CUDA helper functions
#include <helper_cuda.h>         // helper functions for CUDA error check

#define MAX_EPSILON_ERROR 5e-3f

#define SCALED_WIDTH 64
#define SCALED_HEIGHT 64

#define SCALES 33


// Define the files that are to be save and the reference images for validation
const char *imageFilename = "lena_bw.pgm";
const char *refFilename   = "ref_rotated.pgm";

const char *sampleName = "simpleTexture";

////////////////////////////////////////////////////////////////////////////////
// Constants
//const float scale = 0.8f;

__constant__ float d_scale_factors[512];
__constant__ uint8_t d_scales;

// Texture reference for 2D float texture
texture<float, 2, cudaReadModeElementType> tex;

// Auto-Verification Code
bool testResult = true;

////////////////////////////////////////////////////////////////////////////////
//! Resize an image using texture lookups
//! @param outputData  output data in global memory
////////////////////////////////////////////////////////////////////////////////

//__global__ void transformKernel(float *outputData,
//                                int width,
//                                int height
//                                )
//{
//	int scaled_width = blockDim.x * gridDim.x;
//	int scaled_height = blockDim.y * gridDim.y;
//	float upper;
//	float lower;
//
//    unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
//    unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
//
//    for (int i=0 ; i<d_scales; i++) {
//    // calculate normalized texture coordinates
//    	upper = (d_scale_factors[i] + 1) * 0.5;
//    	lower = (1 - d_scale_factors[i]) * 0.5;
//        outputData[y * scaled_width + x + i*scaled_width*scaled_height] = tex2D(tex,
//        					(float) x * (upper-lower) / scaled_width +lower, (float) y * (upper-lower) / scaled_height +lower);
//	}
//}

__global__ void transformKernel(float *outputData, int i)
{
    unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

    // calculate normalized texture coordinates TODO pre-compute (LUT ?)
	int scaled_width = blockDim.x * gridDim.x;
	int scaled_height = blockDim.y * gridDim.y;
	float upper;
	float lower;
	upper = (d_scale_factors[i] + 1) * 0.5;
	lower = (1 - d_scale_factors[i]) * 0.5;

	outputData[y * scaled_width + x] = tex2D(tex,
			(float) x * (upper-lower) / scaled_width +lower, (float) y * (upper-lower) / scaled_height +lower);
}


////////////////////////////////////////////////////////////////////////////////
// Declaration, forward
void runTest(int argc, char **argv);

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
    printf("%s starting...\n", sampleName);

    // Process command-line arguments
    if (argc > 1)
    {
        if (checkCmdLineFlag(argc, (const char **) argv, "input"))
        {
            getCmdLineArgumentString(argc,
                                     (const char **) argv,
                                     "input",
                                     (char **) &imageFilename);

            if (checkCmdLineFlag(argc, (const char **) argv, "reference"))
            {
                getCmdLineArgumentString(argc,
                                         (const char **) argv,
                                         "reference",
                                         (char **) &refFilename);
            }
            else
            {
                printf("-input flag should be used with -reference flag");
                exit(EXIT_FAILURE);
            }
        }
        else if (checkCmdLineFlag(argc, (const char **) argv, "reference"))
        {
            printf("-reference flag should be used with -input flag");
            exit(EXIT_FAILURE);
        }
    }

    runTest(argc, argv);

    printf("%s completed, returned %s\n",
           sampleName,
           testResult ? "OK" : "ERROR!");
    exit(testResult ? EXIT_SUCCESS : EXIT_FAILURE);
}

////////////////////////////////////////////////////////////////////////////////
//! Run a simple test for CUDA
////////////////////////////////////////////////////////////////////////////////
void runTest(int argc, char **argv)
{
    int devID = findCudaDevice(argc, (const char **) argv);

    // load image from disk
    float *hData = NULL;
    unsigned int width, height;
    char *imagePath = sdkFindFilePath(imageFilename, argv[0]);

    if (imagePath == NULL)
    {
        printf("Unable to source image file: %s\n", imageFilename);
        exit(EXIT_FAILURE);
    }

    sdkLoadPGM(imagePath, &hData, &width, &height);

    unsigned int size = width * height * sizeof(float);
    unsigned int scaled_width = SCALED_WIDTH; //floor(scale * width);
    unsigned int scaled_height = SCALED_HEIGHT; //floor(scale * height);
    unsigned int scaled_width_size = scaled_width * sizeof(float);
    unsigned int scaled_height_size = scaled_height * sizeof(float);

    float scale_step = 1.02;
    uint8_t scales = SCALES;
    float scale_factors[scales];

    for (int i=0; i<scales ; i++) {
    	scale_factors[i] = pow(scale_step, i+1);
    }

    for (int i=0; i<scales ; i++) {
    	scale_factors[i] = scale_factors[i] / scale_factors[scales-1];
    }

    printf("Loaded '%s', %d x %d pixels\n", imageFilename, width, height);

    //Load reference image from image (output)
    float *hDataRef = (float *) malloc(size);
    char *refPath = sdkFindFilePath(refFilename, argv[0]);

    if (refPath == NULL)
    {
        printf("Unable to find reference image file: %s\n", refFilename);
        exit(EXIT_FAILURE);
    }

    sdkLoadPGM(refPath, &hDataRef, &width, &height);

    checkCudaErrors(cudaMemcpyToSymbol(d_scale_factors, scale_factors, scales*sizeof(float)));
    checkCudaErrors(cudaMemcpyToSymbol(d_scales, &scales, sizeof(uint8_t)));

    // Allocate device memory for result
    float *dData = NULL;
    checkCudaErrors(cudaMalloc((void **) &dData, scaled_width_size * scaled_height_size * scales));

    // Allocate array and copy image data
    cudaChannelFormatDesc channelDesc =
        cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
    cudaArray *cuArray;
    checkCudaErrors(cudaMallocArray(&cuArray,
                                    &channelDesc,
                                    width,
                                    height));
    checkCudaErrors(cudaMemcpyToArray(cuArray,
                                      0,
                                      0,
                                      hData,
                                      size,
                                      cudaMemcpyHostToDevice));

    // Set texture parameters
    tex.addressMode[0] = cudaAddressModeClamp;
    tex.addressMode[1] = cudaAddressModeClamp;
    tex.filterMode = cudaFilterModeLinear;
    tex.normalized = true;    // access with normalized texture coordinates

    // Bind the array to the texture
    checkCudaErrors(cudaBindTextureToArray(tex, cuArray, channelDesc));

    dim3 dimBlock(8, 8, 1);
    dim3 dimGrid(scaled_width / dimBlock.x, scaled_height / dimBlock.y, 1);

    uint8_t nStreams = scales;
    int streamSize = scaled_width * scaled_height;
    cudaStream_t streams[nStreams];
    for (int i = 0; i < nStreams; ++i) {
    	cudaStreamCreate(&streams[i]);
	}

    // Allocate mem for the result on host side
//    float *hOutputData = (float *) malloc(scaled_width_size * scaled_height_size * scales);
    float *hOutputData;
    checkCudaErrors(cudaMallocHost((void **)&hOutputData, scaled_width_size * scaled_height_size * scales));

    // Warmup
//    transformKernel<<<dimGrid, dimBlock, 0>>>(dData, width, height);

    checkCudaErrors(cudaDeviceSynchronize());
    StopWatchInterface *timer = NULL;
    sdkCreateTimer(&timer);
    sdkStartTimer(&timer);

    // Execute the kernel
//    transformKernel<<<dimGrid, dimBlock, 0>>>(dData, width, height);
//
//    // copy result from device to host
//    checkCudaErrors(cudaMemcpy(hOutputData,
//                               dData,
//                               scaled_width_size * scaled_height_size * scales,
//                               cudaMemcpyDeviceToHost));

    for (int i = 0; i < nStreams; ++i) {
		int offset = i * streamSize;
		transformKernel<<<dimGrid, dimBlock, 0, streams[i]>>>(&dData[offset], i);
		checkCudaErrors(cudaMemcpyAsync(&hOutputData[offset], &dData[offset], scaled_width_size * scaled_height_size,
				cudaMemcpyDeviceToHost, streams[i]));
	}

    // Check if kernel execution generated an error
    getLastCudaError("Kernel execution failed");

    checkCudaErrors(cudaDeviceSynchronize());
    sdkStopTimer(&timer);
    printf("Processing time: %f (ms)\n", sdkGetTimerValue(&timer));
    printf("%.2f Mpixels/sec\n",
           (width *height / (sdkGetTimerValue(&timer) / 1000.0f)) / 1e6);
    sdkDeleteTimer(&timer);


    // Write result to file
    char outputFilename[1024];
    char suffix[10];
    strcpy(outputFilename, imagePath);
	for (int i = 0; i < scales; ++i) {
		sprintf(suffix, "%d", i);
		strcpy(outputFilename + strlen(imagePath) - 4, suffix);
		strcat(outputFilename + strlen(outputFilename) - 1, "_out.pgm");
		sdkSavePGM(outputFilename, hOutputData + scaled_width * scaled_height * i, scaled_width, scaled_height);
//		printf("Wrote '%s'\n", outputFilename);
	}

	int blocks[2];
	int res_dimy = 512;
	int res_dimx = 512;
	blocks[0] = (int)round((float)res_dimy/8.0);
	blocks[1] = (int)round((float)res_dimx/8.0);

	int hy = max(blocks[0]-2, 0);
	int hx = max(blocks[1]-2, 0);
	int hz = 31;

	int sbin = 8;

	int eleSize = hx*hy*hz;

#define MAX_ALLOC 300
#define MAX_BINS 18
#define MAX_DIMS 32

    float* d_pHist;
    float* d_pNorm;
    float* d_pOut;
    float* h_pDescriptor;

    h_pDescriptor = (float*)malloc(sizeof(float) * hx*hy*hz);

    cudaMalloc((void**)&d_pHist, MAX_ALLOC * MAX_ALLOC * MAX_BINS * sizeof(float));
	cudaMemset(d_pHist, 0, MAX_ALLOC * MAX_ALLOC * MAX_BINS * sizeof(float));

	// blocks outputs
	//const int nBlocks = MAX_IMAGE_DIMENSION/8 * MAX_IMAGE_DIMENSION/8 ;	// WE ASSUME MAXIMUM IMAGE SIZE OF 1280x1280
	//const int blocksMemorySize = nBlocks * HOG_BLOCK_CELLS_X * HOG_BLOCK_CELLS_Y * NBINS * sizeof(float);
	cudaMalloc((void**)&d_pNorm, MAX_ALLOC * MAX_ALLOC * sizeof(float));
    cudaMemset(d_pNorm, 0, MAX_ALLOC * MAX_ALLOC * sizeof(float));

    cudaMalloc((void**)&d_pOut, MAX_ALLOC * MAX_ALLOC * MAX_DIMS * sizeof(float));
	cudaMemset(d_pOut, 0, MAX_ALLOC * MAX_ALLOC * MAX_DIMS * sizeof(float));

    if( voc_prepare_image2(h_pImg, width, height, sbin) ) {
		printf("prepare_image failed\n");
	}

	int res = voc_compute_gradients(width, height, sbin, blocks[0], blocks[1], d_pHist);
	if( res ) printf("compute_gradients failed: %d\n", res);

	res = voc_compute_block_energy(blocks[0], blocks[1], d_pHist, d_pNorm);
    if( res ) printf("compute_block_energy failed: %d\n", res);

    voc_compute_features(blocks[0], blocks[1], d_pHist, d_pNorm, d_pOut);

    cudaMemcpy(h_pDescriptor, d_pOut, sizeof(float) * eleSize, cudaMemcpyDeviceToHost);

	cudaFree(d_pHist); d_pHist = NULL;
	cudaFree(d_pNorm); d_pNorm = NULL;
	cudaFree(d_pOut); d_pOut = NULL;
    free(h_pDescriptor);

//    // Write regression file if necessary
//    if (checkCmdLineFlag(argc, (const char **) argv, "regression"))
//    {
//        // Write file for regression test
//        sdkWriteFile<float>("./data/regression.dat",
//                            hOutputData,
//                            width*height,
//                            0.0f,
//                            false);
//    }
//    else
//    {
//        // We need to reload the data from disk,
//        // because it is inverted upon output
//        sdkLoadPGM(outputFilename, &hOutputData, &width, &height);
//
//        printf("Comparing files\n");
//        printf("\toutput:    <%s>\n", outputFilename);
//        printf("\treference: <%s>\n", refPath);
//
//        testResult = compareData(hOutputData,
//                                 hDataRef,
//                                 width*height,
//                                 MAX_EPSILON_ERROR,
//                                 0.15f);
//    }

    checkCudaErrors(cudaFree(dData));
    checkCudaErrors(cudaFreeArray(cuArray));
    checkCudaErrors(cudaFreeHost(hOutputData));
    free(imagePath);
    free(refPath);
}
