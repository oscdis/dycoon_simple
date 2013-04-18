
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <math.h>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#include <string>
#include <fstream>
#include <vector>

//#include <windows.h>

//#pragma comment( lib, "winmm.lib" )

typedef struct{
	float x;
	float y;
	float z;
} Vec3;

typedef long DWORD;
#include <sys/time.h>

long timeGetTime(){
	struct timeval tv;
	gettimeofday(&tv, NULL);
	return tv.tv_sec * 1000 + tv.tv_usec / 1000;
}

bool LoadString(const std::string & fileName, std::string & data)
{
	std::ifstream f(fileName.c_str());

	if(f.bad())
	{
		return false;
	}

	size_t fileSize = (size_t)f.seekg(0, std::ios::end).tellg();
	f.seekg(0, std::ios::beg);

	//printf("fileSize %d\n", (int)fileSize);

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
		printf("error %s %d %d\n", __FILE__, __LINE__, status);\
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

	//	printf("OpenCL device ID %d\n", (int)devices[0]);

	cl_device_type type = 0;
	status = clGetDeviceInfo(devices[0],
			CL_DEVICE_TYPE,
			sizeof(cl_device_type),
			&type,
			NULL);
	CL_STATUS_CKECK();

	printf("OpenCL device type %d\n", type);

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
	static cl_mem a;
	
	unsigned int asz = 100000000;

	a = clCreateBuffer(
				context,
				CL_MEM_WRITE_ONLY,
				//sizeof(Vec3) * asz,
				sizeof(int) * asz,
				NULL,
				&status);
	CL_STATUS_CKECK();

	size_t jobSize = 5000;
	size_t globalWorkSize = asz / jobSize;

	static cl_mem ct0;
	static cl_mem ct1;

	ct0 = clCreateBuffer(
				context,
				CL_MEM_READ_WRITE,
				sizeof(int) * globalWorkSize,
				NULL,
				&status);
	CL_STATUS_CKECK();

	ct1 = clCreateBuffer(
				context,
				CL_MEM_READ_WRITE,
				sizeof(int) * globalWorkSize,
				NULL,
				&status);
	CL_STATUS_CKECK();

	int * sourceA;
	sourceA = new int[asz];

	unsigned int i, isz;

	isz = asz;
	for(i = 0 ; i < isz ; i++)
	{
		sourceA[i] = i;
	}

	static int * dstCt0;
	static int * dstCt1;
	dstCt0 = new int[globalWorkSize]();
	dstCt1 = new int[globalWorkSize]();

	isz = (int)globalWorkSize;
	for(i = 0 ; i < isz ; i++)
	{
		dstCt0[i] = 0;
		dstCt1[i] = 0;
	}

	//
	static cl_program program;
	const char * source;
	/*
		"__kernel void Count(__global const int * a, __global int * ct0, __global int * ct1)\n"
		"{\n"
		"	int gid = get_global_id(0);\n"
		"	int i;\n"
		"	int isz;\n"
		"	int o = gid * 10000;\n"
		"	int ct0t = 0;\n"
		"	int ct1t = 0;\n"
		"	isz = o + 10000;\n"
		"	for(i = o ; i <isz ; i++)\n"
		"	{\n"
		"		if((a[i] % 2) == 0)\n"
		"		{\n"
		"			ct0t++;\n"
		"		}\n"
		"		else\n"
		"		{\n"
		"			ct1t++;\n"
		"		}\n"
		"	}\n"
		"	ct0[gid] = ct0t;\n"
		"	ct1[gid] = ct1t;\n"
		"}\n"
		;
	//*/

	std::string str;
	LoadString("simple.cl", str);
	source = str.c_str();

	size_t slen = strlen(source);
	//printf("slen %d\n", (int)slen);
	//printf("str.c_str() %s\n", source);
	program = clCreateProgramWithSource(context, 1, (const char **)&source, &slen, &status);
	CL_STATUS_CKECK();

	status = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
	if(status != CL_SUCCESS)
	{
		std::vector<char> str;
		str.resize(0x10000);
		clGetProgramBuildInfo(program, devices[0],
			CL_PROGRAM_BUILD_LOG,
			str.size(),
			&str[0],
			NULL);
		
		printf("status %d\n", status);
		printf("error log\n%s\n", &str[0]);
		printf("error %s %d\n", __FILE__, __LINE__);
		exit(1);
	}
	CL_STATUS_CKECK();

	//
	static cl_kernel kernel;
	kernel = clCreateKernel(program, "Count", &status);
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

	status = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&ct0);
	CL_STATUS_CKECK();

	status = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&ct1);
	CL_STATUS_CKECK();



	//

	DWORD tm;
	cl_event eve;

	tm = timeGetTime();

	status = clEnqueueWriteBuffer(commandQueue, a, CL_FALSE, 0, sizeof(int) * asz, sourceA, 0, NULL, &eve);
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
	localWorkSize = 16;
	status = clEnqueueNDRangeKernel(commandQueue, kernel, 1, NULL, &globalWorkSize, &localWorkSize, 0, NULL, NULL);
	CL_STATUS_CKECK();


	//

	status = clEnqueueReadBuffer(commandQueue, ct0, CL_TRUE, 0, sizeof(int) * globalWorkSize, dstCt0, 0, NULL, NULL);
	if(status != CL_SUCCESS)
	CL_STATUS_CKECK();

	status = clEnqueueReadBuffer(commandQueue, ct1, CL_TRUE, 0, sizeof(int) * globalWorkSize, dstCt1, 0, NULL, NULL);
	if(status != CL_SUCCESS)
	CL_STATUS_CKECK();

	//
	printf("OpenCL result\n");

	int sumCt0 = 0;
	int sumCt1 = 0;

	isz = (int)globalWorkSize;
	for(i = 0 ; i < isz ; i++)
	{
		sumCt0 += dstCt0[i];
		sumCt1 += dstCt1[i];
	}

	printf("ct0 = %d\n", sumCt0);
	printf("ct1 = %d\n", sumCt1);

	tm = timeGetTime() - tm;
	printf("time = %d\n", (int)tm);

	//
	printf("CPU result\n");

	tm = timeGetTime();

	sumCt0 = 0;
	sumCt1 = 0;

	isz = asz;
	for(i = 0 ; i < isz ; i++)
	{
		if((sourceA[i] % 2) == 0)
		{
			sumCt0++;
		}
		else
		{
			sumCt1++;
		}
	}

	printf("ct0 = %d\n", sumCt0);
	printf("ct1 = %d\n", sumCt1);

	tm = timeGetTime() - tm;
	printf("time = %d\n", (int)tm);


	//
	clReleaseKernel(kernel);
	clReleaseProgram(program);

	clReleaseMemObject(a);
	clReleaseMemObject(ct0);
	clReleaseMemObject(ct1);

	clReleaseCommandQueue(commandQueue);

	clReleaseContext(context);

	//
	delete[] sourceA;
	delete[] dstCt0;
	delete[] dstCt1;

	return 0;
}
