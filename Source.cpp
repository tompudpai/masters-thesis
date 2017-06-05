#include <stdio.h>
#include <stdlib.h>
#include <Windows.h>
#include <math.h>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#define PLATFORM_ID 0
#define DEVICE_ID 0
#define DEVICE_TYPE CL_DEVICE_TYPE_GPU
#define LOCAL 1
#define LOAD_PARALLEL 1
#define LOOP_UNROLLING 1
#define GLOBAL_ITEM_SIZE_X 512
#define GLOBAL_ITEM_SIZE_Y 512
#define LOCAL_ITEM_SIZE_X 512
#define LOCAL_ITEM_SIZE_Y 1
#define NUM_WORKGROUPS_X GLOBAL_ITEM_SIZE_X / LOCAL_ITEM_SIZE_X

#define MAX_SOURCE_SIZE (0x800000)
#define DISPARITY_LEVEL 64
#define WINDOW_RADIUS 2
#define IM_WIDTH 434
#define IM_HEIGHT 383
#define PADDED_HEIGHT (int)pow(2, ceil(log(IM_HEIGHT)/log(2)))
#define PADDED_WIDTH (int)pow(2, ceil(log(IM_WIDTH)/log(2)))
#define PADDED_SIZE PADDED_WIDTH * PADDED_HEIGHT
#define LOCAL_WIDTH 512
#define LOCAL_HEIGHT (int)(2 * WINDOW_RADIUS + 1)
#define LOCAL_SIZE LOCAL_WIDTH * LOCAL_HEIGHT


int main()
{
	printf("%d x %d input images\n", IM_HEIGHT, IM_WIDTH);
	printf("%d x %d padded dimensions\n", PADDED_HEIGHT, PADDED_WIDTH);
	printf("%d x %d window\n", LOCAL_HEIGHT, LOCAL_HEIGHT);
	printf("Disparity Level: %d\n", DISPARITY_LEVEL);
	printf("Number of workgroups(x): %d\n", NUM_WORKGROUPS_X);
	if (LOCAL) {
		printf("Running local implementation");
		if (LOAD_PARALLEL)
			printf(" in parallel");
		else
			printf(" not in parallel");
	}
	else
		printf("Running global implementation");
	if (LOOP_UNROLLING)
		printf(" with loop unrolling.\n\n");
	else
		printf(" without loop unrolling.\n\n");

	LARGE_INTEGER perfFrequency;
	LARGE_INTEGER performanceCountNDRangeStart;
	LARGE_INTEGER performanceCountNDRangeStop;

	FILE *fp;
	char fileName[] = "./sad.cl";
	char *source_str;
	size_t source_size;

	/* Load the source code containing the kernel*/
	fp = fopen(fileName, "r");
	if (!fp) {
		fprintf(stderr, "Failed to load kernel.\n");
		exit(1);
	}
	source_str = (char*)malloc(MAX_SOURCE_SIZE);
	source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp);
	fclose(fp);

	/* Get Platform ID and Info */
	cl_int ret;
	cl_uint ret_num_platforms;
	cl_platform_id *platforms = NULL;
	char* platformInfo;
	size_t platformInfoSize;

	// get platform count
	ret = clGetPlatformIDs(3, NULL, &ret_num_platforms);
	printf("Number of platforms: %d\n", ret_num_platforms);

	// get all platforms
	platforms = new cl_platform_id[ret_num_platforms];
	ret = clGetPlatformIDs(ret_num_platforms, platforms, NULL);

	// get platform name value size
	clGetPlatformInfo(platforms[PLATFORM_ID], CL_PLATFORM_NAME, 0, NULL, &platformInfoSize);
	platformInfo = (char*)malloc(platformInfoSize);

	// get platform name value
	clGetPlatformInfo(platforms[PLATFORM_ID], CL_PLATFORM_NAME, platformInfoSize, platformInfo, NULL);

	printf("%-11s:\n %d. %s\n\n", "Platform chosen", PLATFORM_ID + 1, platformInfo);
	free(platformInfo);

	/* Get Device ID and Info */
	cl_uint ret_num_devices;
	cl_device_id *devices = NULL;

	//get device count
	ret = clGetDeviceIDs(platforms[PLATFORM_ID], DEVICE_TYPE, 1, devices, &ret_num_devices);
	printf("Number of devices: %d\n", ret_num_devices);

	//get all devices
	devices = new cl_device_id[ret_num_devices];
	ret = clGetDeviceIDs(platforms[PLATFORM_ID], DEVICE_TYPE, ret_num_devices, devices, NULL);

	//print device names
	char* deviceInfo;
	size_t deviceInfoSize, local_size_size;
	cl_ulong local_size;
	cl_int cl_local_size;

	// get device name value size
	clGetDeviceInfo(devices[DEVICE_ID], CL_DEVICE_NAME, 0, NULL, &deviceInfoSize);
	deviceInfo = (char*)malloc(deviceInfoSize);

	// get device name value
	clGetDeviceInfo(devices[DEVICE_ID], CL_DEVICE_NAME, deviceInfoSize, deviceInfo, NULL);

	printf("%-11s:\n %d. %s\n\n", "Device chosen", DEVICE_ID + 1, deviceInfo);
	free(deviceInfo);

	clGetDeviceInfo(devices[DEVICE_ID], CL_DEVICE_LOCAL_MEM_SIZE, sizeof(local_size), &local_size, &local_size_size);
	printf("CL_DEVICE_LOCAL_MEM_SIZE = %d\n___________________\n\n", (int)local_size);

	/* Create OpenCL context */
	cl_context context = NULL;
	context = clCreateContext(NULL, 1, &devices[DEVICE_ID], NULL, NULL, &ret);

	/* Create Command Queue */
	cl_command_queue command_queue = NULL;
	command_queue = clCreateCommandQueue(context, devices[DEVICE_ID], CL_QUEUE_PROFILING_ENABLE, &ret);

	/* Create Memory Buffer */
	cl_mem Rmobj = NULL;
	cl_mem Lmobj = NULL;
	cl_mem SADmobj = NULL;
	cl_mem imdimmobj = NULL;
	cl_mem paddeddimmobj = NULL;
	cl_mem firstmobj = NULL;
	cl_mem localmobj = NULL;
	cl_mem locsizemobj = NULL;
	cl_mem unrollmobj = NULL;
	cl_mem dispmobj = NULL;
	cl_mem radmobj = NULL;
	cl_mem rowdonemobj = NULL;

	unsigned char *R;
	unsigned char *L;
	int *SAD;
	int *imdim;
	int *paddeddim;
	bool *first;
	bool *local;
	int *locsize;
	bool *unroll;
	int *disp;
	int *rad;
	bool *rowDone;

	R = (unsigned char *)malloc(PADDED_SIZE * sizeof(unsigned char));
	L = (unsigned char *)malloc(PADDED_SIZE * sizeof(unsigned char));
	SAD = (int *)malloc(PADDED_SIZE * sizeof(int));
	imdim = (int*)malloc(2 * sizeof(int));
	paddeddim = (int*)malloc(2 * sizeof(int));
	first = (bool*)malloc(GLOBAL_ITEM_SIZE_Y * NUM_WORKGROUPS_X * sizeof(bool));
	local = (bool*)malloc(2 * sizeof(bool));
	locsize = (int*)malloc(2 * sizeof(int));
	unroll = (bool*)malloc(sizeof(bool));
	disp = (int*)malloc(sizeof(int));
	rad = (int*)malloc(sizeof(int));
	rowDone = (bool*)malloc(GLOBAL_ITEM_SIZE_Y * NUM_WORKGROUPS_X * sizeof(bool));

	Rmobj = clCreateBuffer(context, CL_MEM_READ_ONLY, PADDED_SIZE * sizeof(unsigned char), NULL, &ret);
	Lmobj = clCreateBuffer(context, CL_MEM_READ_ONLY, PADDED_SIZE * sizeof(unsigned char), NULL, &ret);
	SADmobj = clCreateBuffer(context, CL_MEM_WRITE_ONLY, PADDED_SIZE * sizeof(int), NULL, &ret);
	imdimmobj = clCreateBuffer(context, CL_MEM_READ_ONLY, 2 * sizeof(int), NULL, &ret);
	paddeddimmobj = clCreateBuffer(context, CL_MEM_READ_ONLY, 2 * sizeof(int), NULL, &ret);
	firstmobj = clCreateBuffer(context, CL_MEM_READ_WRITE, GLOBAL_ITEM_SIZE_Y * NUM_WORKGROUPS_X * sizeof(bool), NULL, &ret);
	localmobj = clCreateBuffer(context, CL_MEM_READ_ONLY, 2 * sizeof(bool), NULL, &ret);
	locsizemobj = clCreateBuffer(context, CL_MEM_READ_ONLY, 2 * sizeof(int), NULL, &ret);
	unrollmobj = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(bool), NULL, &ret);
	dispmobj = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(int), NULL, &ret);
	radmobj = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(int), NULL, &ret);
	rowdonemobj = clCreateBuffer(context, CL_MEM_READ_WRITE, GLOBAL_ITEM_SIZE_Y * NUM_WORKGROUPS_X * sizeof(bool), NULL, &ret);


	/* Load Input DATA */
	FILE *fpL, *fpR;
	fpL = fopen("venusL_bw.dat", "r");
	if (fpL == NULL)
		return -1;
	printf("imgL.txt opened |");

	fpR = fopen("venusR_bw.dat", "r");
	if (fpR == NULL)
		return -1;
	printf(" imgR.txt opened\n\n");

	int r, l, i, j;

	fscanf(fpR, "%d", &r);
	fscanf(fpL, "%d", &l);
	for (i = 0; i < PADDED_HEIGHT; i++) {
		for (j = 0; j < PADDED_WIDTH; j++) {
			if (!feof(fpR) || !feof(fpL))
			{
				if (i < IM_HEIGHT && j < IM_WIDTH)
				{
					R[i * PADDED_WIDTH + j] = (unsigned char) r;
					L[i * PADDED_WIDTH + j] = (unsigned char) l;
					fscanf(fpR, "%d", &r);
					fscanf(fpL, "%d", &l);
				}
				else {
					R[i * PADDED_WIDTH + j] = 0;
					L[i * PADDED_WIDTH + j] = 0;
				}
				//printf("%d %d\n", R[i * PADDED_WIDTH + j], L[i * PADDED_WIDTH + j]);
			}
		}
	}


	imdim[0] = IM_HEIGHT;
	imdim[1] = IM_WIDTH;
	paddeddim[0] = PADDED_HEIGHT;
	paddeddim[1] = PADDED_WIDTH;
	for (int f = 0; f < NUM_WORKGROUPS_X; f++)
		for (int g = 0; g < GLOBAL_ITEM_SIZE_Y; g++)
		{
			first[g * NUM_WORKGROUPS_X + f] = false;
			/*if (g < WINDOW_RADIUS)
				rowDone[g * NUM_WORKGROUPS_X + f] = true;
			else*/
			rowDone[g * NUM_WORKGROUPS_X + f] = false;
		}
	local[0] = LOCAL;
	local[1] = LOAD_PARALLEL;
	locsize[0] = LOCAL_ITEM_SIZE_Y;
	locsize[1] = LOCAL_ITEM_SIZE_X;
	unroll[0] = LOOP_UNROLLING;
	disp[0] = DISPARITY_LEVEL;
	rad[0] = WINDOW_RADIUS;

	/* Write disparity values to output file */
	FILE *fpDispL, *fpDispR;
	int x, y;
	fpDispL = fopen("cDispL.dat", "w");
	fpDispR = fopen("cDispR.dat", "w");
	for (x = 0; x < IM_HEIGHT; x++) {
		for (y = 0; y < IM_WIDTH; y++) {
			fprintf(fpDispL, "%i ", L[x * PADDED_WIDTH + y]);
			fprintf(fpDispR, "%i ", R[x * PADDED_WIDTH + y]);
		}
		fprintf(fpDispL, "\n");
		fprintf(fpDispR, "\n");
	}
	printf("fpDispL.dat written |");
	printf("  fpDispR.dat written\n\n");

	/* Copy input data to mem buffer */
	ret = clEnqueueWriteBuffer(command_queue, Rmobj, CL_TRUE, 0, PADDED_SIZE * sizeof(unsigned char), R,
		0, NULL, NULL);
	ret = clEnqueueWriteBuffer(command_queue, Lmobj, CL_TRUE, 0, PADDED_SIZE * sizeof(unsigned char), L,
		0, NULL, NULL);
	ret = clEnqueueWriteBuffer(command_queue, imdimmobj, CL_TRUE, 0, 2 * sizeof(int), imdim,
		0, NULL, NULL);
	ret = clEnqueueWriteBuffer(command_queue, paddeddimmobj, CL_TRUE, 0, 2 * sizeof(int), paddeddim,
		0, NULL, NULL);
	ret = clEnqueueWriteBuffer(command_queue, firstmobj, CL_TRUE, 0, GLOBAL_ITEM_SIZE_Y * NUM_WORKGROUPS_X * sizeof(bool), first,
		0, NULL, NULL);
	ret = clEnqueueWriteBuffer(command_queue, localmobj, CL_TRUE, 0, 2 * sizeof(bool), local,
		0, NULL, NULL);
	ret = clEnqueueWriteBuffer(command_queue, locsizemobj, CL_TRUE, 0, 2 * sizeof(int), locsize,
		0, NULL, NULL);
	ret = clEnqueueWriteBuffer(command_queue, unrollmobj, CL_TRUE, 0, sizeof(bool), unroll,
		0, NULL, NULL);
	ret = clEnqueueWriteBuffer(command_queue, dispmobj, CL_TRUE, 0, sizeof(int), disp,
		0, NULL, NULL);
	ret = clEnqueueWriteBuffer(command_queue, radmobj, CL_TRUE, 0, sizeof(int), rad,
		0, NULL, NULL);
	ret = clEnqueueWriteBuffer(command_queue, rowdonemobj, CL_TRUE, 0, GLOBAL_ITEM_SIZE_Y * NUM_WORKGROUPS_X * sizeof(bool), rowDone,
		0, NULL, NULL);

	/* Create Kernel Program from the source */
	cl_program program = NULL;
	program = clCreateProgramWithSource(context, 1, (const char **)&source_str,
		(const size_t *)&source_size, &ret);

	/* Build Kernel Program */
	ret = clBuildProgram(program, 1, &devices[DEVICE_ID], NULL, NULL, NULL);

	/* Create OpenCL Kernel */
	cl_kernel kernel[2] = { NULL, NULL };
	kernel[0] = clCreateKernel(program, "copyToLocal", &ret);
	kernel[1] = clCreateKernel(program, "sad", &ret);


	/* Set OpenCL Kernel Parameters */
	cl_local_size = local_size / 4;

	ret = clSetKernelArg(kernel[1], 0, sizeof(cl_mem), (void *)&Rmobj);
	ret = clSetKernelArg(kernel[1], 1, sizeof(cl_mem), (void *)&Lmobj);
	ret = clSetKernelArg(kernel[1], 2, sizeof(cl_mem), (void *)&SADmobj);
	ret = clSetKernelArg(kernel[1], 3, sizeof(cl_mem), (void *)&imdimmobj);
	ret = clSetKernelArg(kernel[1], 4, sizeof(cl_mem), (void *)&paddeddimmobj);
	ret = clSetKernelArg(kernel[1], 5, sizeof(cl_mem), (void *)&firstmobj);
	ret = clSetKernelArg(kernel[1], 6, LOCAL_SIZE * sizeof(unsigned char), NULL);
	ret = clSetKernelArg(kernel[1], 7, LOCAL_SIZE * sizeof(unsigned char), NULL);
	ret = clSetKernelArg(kernel[1], 8, sizeof(cl_local_size), &cl_local_size); //currently unused
	ret = clSetKernelArg(kernel[1], 9, sizeof(cl_mem), (void *)&localmobj);
	ret = clSetKernelArg(kernel[1], 10, sizeof(cl_mem), (void *)&locsizemobj);
	ret = clSetKernelArg(kernel[1], 11, sizeof(cl_mem), (void *)&unrollmobj);
	ret = clSetKernelArg(kernel[1], 12, sizeof(cl_mem), (void *)&dispmobj);
	ret = clSetKernelArg(kernel[1], 13, sizeof(cl_mem), (void *)&radmobj);
	ret = clSetKernelArg(kernel[1], 14, sizeof(cl_mem), (void *)&rowdonemobj);

	//printf("size of int: %d\n", sizeof(int));
	printf("size of local available: %d\n", (int)local_size);
	//printf("size of cl_local_size: %d\n", (int)cl_local_size);
	printf("size of local used: #CUs * %d\n", 2 * LOCAL_SIZE * sizeof(unsigned char));

	bool queueProfilingEnable = true;
	if (queueProfilingEnable)
		QueryPerformanceCounter(&performanceCountNDRangeStart);

	/* Execute OpenCL Kernel */
	//size_t global_item_size[2] = { PADDED_WIDTH, PADDED_HEIGHT };
	//size_t local_item_size[2] = { 128, 64 };
	size_t global_item_size[2] = { GLOBAL_ITEM_SIZE_X, GLOBAL_ITEM_SIZE_Y };
	size_t local_item_size[2] = { LOCAL_ITEM_SIZE_X, LOCAL_ITEM_SIZE_Y };
	cl_uint NDRange_dim = 2;
	cl_event tevt;

	ret = clEnqueueTask(command_queue, kernel[0], 0, NULL, NULL);
	ret = clEnqueueNDRangeKernel(command_queue, kernel[1], NDRange_dim, NULL,
		global_item_size, local_item_size, 0, NULL, &tevt);
	clWaitForEvents(1, &tevt);

	if (ret == CL_OUT_OF_RESOURCES) {
		puts("too large local");
		return 1;
	}

	/* Copy results from the memory buffer */
	ret = clEnqueueReadBuffer(command_queue, SADmobj, CL_TRUE, 0,
		PADDED_SIZE * sizeof(int), SAD, 0, NULL, NULL);

	/* Finalization */
	ret = clFlush(command_queue);
	ret = clFinish(command_queue);

	/* Get profiling data */
	cl_ulong time_start = 0, time_end = 0;
	double total_time;

	clGetEventProfilingInfo(tevt, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
	clGetEventProfilingInfo(tevt, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);
	total_time = time_end - time_start;
	printf("Execution time in milliseconds = %0.4f ms\n", (total_time / 1000000.0));

	if (queueProfilingEnable)
		QueryPerformanceCounter(&performanceCountNDRangeStop);

	/* Write disparity values to output file */
	FILE *fpDisp;
	int h, w;
	fpDisp = fopen("cDisp.dat", "w");
	for (h = 0; h < PADDED_HEIGHT; h++) {
		for (w = 0; w < PADDED_WIDTH; w++) {
			if (h < IM_HEIGHT && w < IM_WIDTH)
				fprintf(fpDisp, "%i ", SAD[h * PADDED_WIDTH + w]);
		}
		if (h < IM_HEIGHT)
			fprintf(fpDisp, "\n");
	}
	printf("cDisp.dat written\n\n");

	// retrieve performance counter frequency
	if (queueProfilingEnable)
	{
		QueryPerformanceFrequency(&perfFrequency);
		printf("NDRange performance counter time %f ms.\n\n",
			1000.0f*(float)(performanceCountNDRangeStop.QuadPart - performanceCountNDRangeStart.QuadPart) / (float)perfFrequency.QuadPart);
	}

	/* Finalization continued */
	ret = clReleaseKernel(kernel[0]);
	ret = clReleaseKernel(kernel[1]);
	ret = clReleaseProgram(program);
	ret = clReleaseMemObject(Rmobj);
	ret = clReleaseMemObject(Lmobj);
	ret = clReleaseMemObject(SADmobj);
	ret = clReleaseMemObject(imdimmobj);
	ret = clReleaseMemObject(paddeddimmobj);
	ret = clReleaseMemObject(firstmobj);
	ret = clReleaseMemObject(localmobj);
	ret = clReleaseMemObject(locsizemobj);
	ret = clReleaseMemObject(unrollmobj);
	ret = clReleaseMemObject(dispmobj);
	ret = clReleaseMemObject(radmobj);
	ret = clReleaseMemObject(rowdonemobj);
	ret = clReleaseCommandQueue(command_queue);
	ret = clReleaseContext(context);
	ret = clReleaseDevice(*devices);

	fclose(fpL);
	fclose(fpR);
	fclose(fpDispL);
	fclose(fpDispR);
	fclose(fpDisp);

	free(platforms);
	free(source_str);
	free(R);
	free(L);
	free(SAD);
	free(imdim);
	free(paddeddim);
	free(first);
	free(local);
	free(locsize);
	free(unroll);
	free(disp);
	free(rad);
	free(rowDone);

	system("pause");

	return 0;
}