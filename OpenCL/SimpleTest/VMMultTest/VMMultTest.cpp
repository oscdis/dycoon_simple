
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <math.h>

// Jens's patch for MacOS
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#include <string>
#include <fstream>
#include <vector>

#include <windows.h>

#pragma comment( lib, "winmm.lib" )

typedef struct{
	float x;
	float y;
	float z;
} Vec3;

typedef struct{
	float _11;
	float _12;
	float _13;
	float _14;

	float _21;
	float _22;
	float _23;
	float _24;

	float _31;
	float _32;
	float _33;
	float _34;

	float _41;
	float _42;
	float _43;
	float _44;

} Mat4x4;

bool LoadString(const std::string & fileName, std::string & data)
{
	std::ifstream f(fileName.c_str());

	if(f.bad())
	{
		return false;
	}

	size_t fileSize = (size_t)f.seekg(0, std::ios::end).tellg();
	f.seekg(0, std::ios::beg);

	std::vector<char> d;
	d.resize(fileSize + 1);
	f.read(&d[0], fileSize);
	d[fileSize] = '\0';

	data = &d[0];

	return true;
}

#define CL_STATUS_CKECK()\
	if(status != CL_SUCCESS)\
	{\
		printf("error %s %d\n", __FILE__, __LINE__);\
		exit(1);\
	}


int main(int argc, char *argv[])
{
	cl_uint numPlatforms;
	cl_platform_id platform = NULL;
	cl_int status;

	int platformIndex = 0;

	if(argc > 0)
	{
		platformIndex = atoi(argv[0]);
	}
	
	status = clGetPlatformIDs(0, NULL, &numPlatforms);
	CL_STATUS_CKECK();

	printf("numPlatforms %d\n", numPlatforms);

	if(platformIndex >= (int)numPlatforms)
	{
		platformIndex = 0;
	}

	static cl_platform_id platforms[512];

	status = clGetPlatformIDs(numPlatforms, platforms, NULL);
	CL_STATUS_CKECK();

	char pbuf[100];
	status = clGetPlatformInfo(platforms[platformIndex],
					CL_PLATFORM_VENDOR,
					sizeof(pbuf),
					pbuf,
					NULL);
	CL_STATUS_CKECK();
	printf("OpenCL platform %s\n", pbuf);

	platform = platforms[platformIndex];

	cl_device_id devices[32];
	cl_uint deviceCount;
	status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 32, devices, &deviceCount);
	CL_STATUS_CKECK();

	printf("OpenCL device ID %d\n", (int)devices[0]);

	cl_device_type type = 0;
	status = clGetDeviceInfo(devices[0],
			CL_DEVICE_TYPE,
			sizeof(type),
			&type,
			NULL);
	CL_STATUS_CKECK();

	printf("OpenCL device type %d\n", type);

	cl_uint maxComputeUnits;
	status = clGetDeviceInfo(devices[0],
			CL_DEVICE_MAX_COMPUTE_UNITS,
			sizeof(maxComputeUnits),
			&maxComputeUnits,
			NULL);

	printf("OpenCL max compute units %d\n", maxComputeUnits);

	//
	cl_context_properties cps[3] = {
		CL_CONTEXT_PLATFORM,
		(cl_context_properties) platform,
		0
	};

	static cl_context context;

	cl_context_properties *cprops = (NULL == platform) ? NULL : cps;
	context = clCreateContext(
			cprops,
			1,
			&devices[0],
			NULL,
			NULL,
			&status);
	CL_STATUS_CKECK();

	//size_t deviceListSize;
    //status = clGetContextInfo(
    //        context,
    //        CL_CONTEXT_DEVICES,
    //        32,
    //        devices,
    //        &deviceListSize);

	static cl_command_queue commandQueue;

	cl_command_queue_properties prop = 0;
	commandQueue = clCreateCommandQueue(
			context,
			devices[0],
			prop,
			&status);
	CL_STATUS_CKECK();

	//
	static cl_mem a, m;
	
	unsigned int asz = 10000000;

	a = clCreateBuffer(
				context,
				CL_MEM_WRITE_ONLY,
				//sizeof(Vec3) * asz,
				sizeof(Vec3) * asz,
				NULL,
				&status);
	CL_STATUS_CKECK();

	m = clCreateBuffer(
				context,
				CL_MEM_WRITE_ONLY,
				sizeof(Mat4x4),
				NULL,
				&status);
	CL_STATUS_CKECK();

	size_t jobSize = 10000;
	size_t globalWorkSize = asz / jobSize;

	static cl_mem r;

	r = clCreateBuffer(
				context,
				CL_MEM_READ_WRITE,
				sizeof(Vec3) * globalWorkSize,
				NULL,
				&status);
	CL_STATUS_CKECK();

	Vec3 * sourceA;
	sourceA = new Vec3[asz];

	unsigned int i, isz;

	isz = asz;
	for(i = 0 ; i < isz ; i++)
	{
		sourceA[i].x = cosf((float)i);
		sourceA[i].y = sinf((float)i);
		sourceA[i].z = sinf(i * 2.0f + 0.5f);
	}

	Mat4x4 ma;

	ma._11 = cosf(0.5f); ma._12 = sinf(0.5f); ma._13 = sinf(0.5f); ma._14 = 0.0f;
	ma._21 = -sinf(0.5f); ma._22 = cosf(0.5f); ma._23 = sinf(0.5f); ma._24 = 0.0f;
	ma._31 = cosf(0.5f); ma._32 = sinf(0.5f); ma._33 = sinf(0.5f); ma._34 = 0.0f;
	ma._41 = 0.1f; ma._42 = 0.2f; ma._43 = 0.3f; ma._44 = 1.0f;

	static Vec3 * dstR;
	dstR = new Vec3[globalWorkSize]();

	isz = (int)globalWorkSize;
	for(i = 0 ; i < isz ; i++)
	{
		dstR[i].x = 0.0f;
		dstR[i].y = 0.0f;
		dstR[i].z = 0.0f;
	}

	//
	static cl_program program;
	const char * source;

	std::string str;
	LoadString("vmmult.cl", str);
	source = str.c_str();

	size_t slen = strlen(source);
	program = clCreateProgramWithSource(context, 1, (const char **)&source, &slen, &status);
	CL_STATUS_CKECK();

	status = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
	CL_STATUS_CKECK();

	//
	static cl_kernel kernel;
	kernel = clCreateKernel(program, "Mult", &status);
	if(status != CL_SUCCESS)
	{
		std::vector<char> str;
		str.resize(0x10000);
		clGetProgramBuildInfo(program, devices[0],
			CL_PROGRAM_BUILD_LOG,
			str.size(),
			&str[0],
			NULL);
		
		printf("error log\n%s\n", &str[0]);
		printf("error %s %d\n", __FILE__, __LINE__);
		exit(1);
	}
	CL_STATUS_CKECK();


	//
	status = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&a);
	CL_STATUS_CKECK();

	status = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&m);
	CL_STATUS_CKECK();

	status = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&r);
	CL_STATUS_CKECK();



	//

	DWORD tm;
	cl_event eve;

	tm = timeGetTime();

	status = clEnqueueWriteBuffer(commandQueue, a, CL_FALSE, 0, sizeof(Vec3) * asz, sourceA, 0, NULL, NULL);
	CL_STATUS_CKECK();

	status = clEnqueueWriteBuffer(commandQueue,m, CL_FALSE, 0, sizeof(Mat4x4), &ma, 0, NULL, &eve);
	CL_STATUS_CKECK();

	//status = clEnqueueTask(commandQueue, kernel, 0, NULL, NULL);
	//if(status != 0)
	//{
	//	printf("error %s %d\n", __FILE__, __LINE__);
	//}

	clWaitForEvents(1, &eve);


	//
	tm = timeGetTime() - tm;
	printf("Write Buffer time = %d\n", (int)tm);

	//
	tm = timeGetTime();


	//
	
	size_t localWorkSize;
	localWorkSize = 8;
	status = clEnqueueNDRangeKernel(commandQueue, kernel, 1, NULL, &globalWorkSize, &localWorkSize, 0, NULL, &eve);
	CL_STATUS_CKECK();

	clWaitForEvents(1, &eve);

	tm = timeGetTime() - tm;
	printf("Run time = %d\n", (int)tm);


	//

	tm = timeGetTime();

	status = clEnqueueReadBuffer(commandQueue, r, CL_TRUE, 0, sizeof(Vec3) * globalWorkSize, dstR, 0, NULL, NULL);
	if(status != CL_SUCCESS)
	CL_STATUS_CKECK();

	//
	printf("OpenCL result\n");

	Vec3 sumR;

	sumR.x = 0.0f;
	sumR.y = 0.0f;
	sumR.z = 0.0f;

	isz = (int)globalWorkSize;
	for(i = 0 ; i < isz ; i++)
	{
		sumR.x += dstR[i].x;
		sumR.y += dstR[i].y;
		sumR.z += dstR[i].z;
	}

	printf("sumR.x = %f\n", sumR.x);
	printf("sumR.y = %f\n", sumR.y);
	printf("sumR.z = %f\n", sumR.z);

	tm = timeGetTime() - tm;
	printf("read time = %d\n", (int)tm);

	//
	printf("CPU result\n");

	tm = timeGetTime();

	sumR.x = 0.0f;
	sumR.y = 0.0f;
	sumR.z = 0.0f;

	isz = asz;
	for(i = 0 ; i < isz ; i++)
	{
		sumR.x += sourceA[i].x * ma._11 + sourceA[i].y * ma._21 + sourceA[i].z * ma._31 + ma._41;
		sumR.y += sourceA[i].x * ma._12 + sourceA[i].y * ma._22 + sourceA[i].z * ma._32 + ma._42;
		sumR.z += sourceA[i].x * ma._13 + sourceA[i].y * ma._23 + sourceA[i].z * ma._33 + ma._43;
	}

	printf("sumR.x = %f\n", sumR.x);
	printf("sumR.y = %f\n", sumR.y);
	printf("sumR.z = %f\n", sumR.z);

	tm = timeGetTime() - tm;
	printf("time = %d\n", (int)tm);


	//
	clReleaseKernel(kernel);
	clReleaseProgram(program);

	clReleaseMemObject(a);
	clReleaseMemObject(m);
	clReleaseMemObject(r);

	clReleaseCommandQueue(commandQueue);

	clReleaseContext(context);

	//
	delete[] sourceA;
	delete[] dstR;


	return 0;
}
