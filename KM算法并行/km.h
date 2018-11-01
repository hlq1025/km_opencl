#pragma once
#include<fstream>
#include <vector>
#include "opencv2/opencv.hpp"
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/calib3d.hpp"
#include<float.h>
#include <CL/cl.h>
//#include "HungarianAlg.h"

class AssignmentProblemSolver
{
private:
	int nx;

	int ny;

	int *visx;

	int *visy;

	int *link;
		
	double *lx;
	
	double *ly;

	double *slack;
	
	double **cost;
	//GPU有关参数
	cl_context context = 0;

	cl_command_queue commandQueue = 0;

	cl_program program = 0;

	cl_device_id device = 0;

	cl_kernel kernel ;

	cl_mem memObjects[6];

	cl_int errNum;

	size_t globalWorkSize[1];


	// --------------------------------------------------------------------------
	// Computes the optimal assignment (minimum overall costs) using Munkres algorithm.
	// --------------------------------------------------------------------------
	int DFS(int x);
	void update(double d);
	//GPU相关函数
	cl_int ConvertToString(const char *pFileName, std::string &Str);
	void GpuInitial();
	cl_context CreateContext();
	cl_command_queue CreateCommandQueue(cl_context context, cl_device_id *device);
	cl_program CreateProgram(cl_context context, cl_device_id device, const char* fileName);
	bool CreateMemObjects(cl_context context, cl_mem memObjects[3]);
	void Cleanup(cl_context context, cl_command_queue commandQueue,

		cl_program program, cl_kernel kernel, cl_mem memObjects[3]);
public:

	AssignmentProblemSolver(int nx,int ny,double **w);
	~AssignmentProblemSolver();
	double solve();
};