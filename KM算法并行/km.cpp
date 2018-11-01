#include "km.h"

using namespace std;

AssignmentProblemSolver::AssignmentProblemSolver(int nx,int ny,double **w)
{
	this->nx = nx;
	
	this->ny = ny;
	
	visx = new int [nx];

	visy=new int [ny];

	link=new int [ny];

	lx=new double[nx];

	ly=new double[ny];

	slack=new double[ny];
	cost = new double*[nx];

	for (int i = 0; i < nx; i++)
	{
		cost[i] = new double[ny];
	}
	for (int i = 0; i < nx;i++)
	for (int j = 0; j < ny; j++)
		cost[i][j] = w[i][j];
	GpuInitial();

}

AssignmentProblemSolver::~AssignmentProblemSolver()
{
}
int AssignmentProblemSolver:: DFS(int x)
{
	visx[x] = 1;
	for (int y = 0; y < ny; y++)
	{
		if (visy[y])
			continue;
		double t = lx[x] + ly[y] - cost[x][y];
		if (abs(t) <= 1e-6)       //
		{
			visy[y] = 1;
			if (link[y] == -1 || DFS(link[y]))
			{
				link[y] = x;
				return 1;
			}
		}
		else if (slack[y] > t)  //不在相等子图中slack 取最小的。//slack[y]表示给定的x值，lx[x]减多少，x和y可以连上。
			slack[y] = t;
	}
	return 0;

}
void AssignmentProblemSolver::update(double d)
{
errNum=	clEnqueueWriteBuffer(commandQueue, memObjects[0], CL_TRUE,



		0, 1 * sizeof(double), &d,



		0, NULL, NULL);

	
errNum|=clEnqueueWriteBuffer(commandQueue, memObjects[1], CL_TRUE,



		0, nx * sizeof(int), visx,



		0, NULL, NULL);

errNum|=clEnqueueWriteBuffer(commandQueue, memObjects[2], CL_TRUE,



		0, ny* sizeof(int), visy,



		0, NULL, NULL);
	
errNum|=clEnqueueWriteBuffer(commandQueue, memObjects[3], CL_TRUE,



		0, ny* sizeof(double), slack,



		0, NULL, NULL);

errNum|=clEnqueueWriteBuffer(commandQueue, memObjects[4], CL_TRUE,



		0, nx* sizeof(double), lx,



		0, NULL, NULL);

errNum|=clEnqueueWriteBuffer(commandQueue, memObjects[5], CL_TRUE,



		0, ny* sizeof(double), ly,



		0, NULL, NULL);

	
errNum |= clEnqueueNDRangeKernel(commandQueue, kernel, 1, NULL,



		globalWorkSize, NULL,



		0, NULL, NULL);


// 六、 读取执行结果并释放OpenCL资源

errNum |= clEnqueueReadBuffer(commandQueue, memObjects[3], CL_TRUE,



		0, ny* sizeof(double), slack,



		0, NULL, NULL);
//for (int i = 0; i < ny; i++)
	//cout << slack[i] << endl;

errNum |= clEnqueueReadBuffer(commandQueue, memObjects[4], CL_TRUE,



	0, nx* sizeof(double), lx,



	0, NULL, NULL);

errNum |= clEnqueueReadBuffer(commandQueue, memObjects[5], CL_TRUE,



	0, ny* sizeof(double), ly,



	0, NULL, NULL);


/*for (int i = 0; i < nx; i++)
	if (visx[i])
		lx[i] -= d;
	for (int i = 0; i < ny; i++)  //修改顶标后，要把所有不在交错树中的Y顶点的slack值都减去d
	{
		if (visy[i])
			ly[i] += d;
		else
			slack[i] -= d;
	}*/


}
double AssignmentProblemSolver::solve()
{
	double start = static_cast<double>(cvGetTickCount());
	int i, j;
	memset(link, -1, ny*sizeof(int));
	memset(ly, 0, ny*sizeof(double));
	for (i = 0; i < nx; i++)            //lx初始化为与它关联边中最大的
	for (j = 0, lx[i] = -DBL_MAX; j < ny; j++)
	if (cost[i][j] > lx[i])
		lx[i] = cost[i][j];
	for (int x = 0; x < nx; x++)
	{
		for (i = 0; i < ny; i++)
			slack[i] = DBL_MAX;
		while (1)
		{
			memset(visx, 0, nx*sizeof(int));
			memset(visy, 0, ny*sizeof(int));
			if (DFS(x))     //若成功（找到了增广轨），则该点增广完成，进入下一个点的增广
				break;  //若失败（没有找到增广轨），则需要改变一些点的标号，使得图中可行边的数量增加。
			//方法为：将所有在增广轨中（就是在增广过程中遍历到）的X方点的标号全部减去一个常数d，
			//所有在增广轨中的Y方点的标号全部加上一个常数d
			double d = DBL_MAX;
			for (i = 0; i < ny; i++)
			if (!visy[i] && d > slack[i])
				d = slack[i];
			update(d);
		}
	}
	double res = 0;
	for (i = 0; i < ny; i++)
	if (link[i] > -1)
		res += cost[link[i]][i];
	double time = ((double)cvGetTickCount() - start) / cvGetTickFrequency();
	cout << "所花费时间为:" << time << "us" << endl;
	return res;
}
cl_int AssignmentProblemSolver::ConvertToString(const char *pFileName, std::string &Str)

{

	size_t		uiSize = 0;

	size_t		uiFileSize = 0;

	char		*pStr = NULL;

	std::fstream fFile(pFileName, (std::fstream::in | std::fstream::binary));

	if (fFile.is_open())

	{

		fFile.seekg(0, std::fstream::end);

		uiSize = uiFileSize = (size_t)fFile.tellg();  // 获得文件大小

		fFile.seekg(0, std::fstream::beg);

		pStr = new char[uiSize + 1];

		if (NULL == pStr)

		{

			fFile.close();

			return 0;

		}

		fFile.read(pStr, uiFileSize);				// 读取uiFileSize字节

		fFile.close();

		pStr[uiSize] = '\0';

		Str = pStr;

		delete[] pStr;

		return 0;

}

//cout << "Error: Failed to open cl file\n:" << pFileName << endl;

return -1;

}
cl_context AssignmentProblemSolver::CreateContext()
{
	cl_int errNum;

	cl_uint numPlatforms;

	cl_platform_id firstPlatformId;

	cl_context context = NULL;



	//选择可用的平台中的第一个

	errNum = clGetPlatformIDs(1, &firstPlatformId, &numPlatforms);

	if (errNum != CL_SUCCESS || numPlatforms <= 0)

	{

		std::cerr << "Failed to find any OpenCL platforms." << std::endl;

		return NULL;

	}

	//创建一个OpenCL上下文环境

	cl_context_properties contextProperties[] =

	{

		CL_CONTEXT_PLATFORM,

		(cl_context_properties)firstPlatformId,

		0

	};

	context = clCreateContextFromType(contextProperties, CL_DEVICE_TYPE_GPU,

		NULL, NULL, &errNum);



	return context;

}
cl_command_queue AssignmentProblemSolver::CreateCommandQueue(cl_context context, cl_device_id *device)

{

	cl_int errNum;

	cl_device_id *devices;

	cl_command_queue commandQueue = NULL;

	size_t deviceBufferSize = -1;

	// 获取设备缓冲区大小

	errNum = clGetContextInfo(context, CL_CONTEXT_DEVICES, 0, NULL, &deviceBufferSize);

	if (deviceBufferSize <= 0)

	{

		std::cerr << "No devices available.";

		return NULL;

	}


	// 为设备分配缓存空间

	devices = new cl_device_id[deviceBufferSize / sizeof(cl_device_id)];

	errNum = clGetContextInfo(context, CL_CONTEXT_DEVICES, deviceBufferSize, devices, NULL);

	//选取可用设备中的第一个

	commandQueue = clCreateCommandQueue(context, devices[0], 0, NULL);

	*device = devices[0];

	delete[] devices;

	return commandQueue;

}
cl_program AssignmentProblemSolver::CreateProgram(cl_context context, cl_device_id device, const char* fileName)

{

	cl_int errNum;

	cl_program program;

	std::ifstream kernelFile(fileName, std::ios::in);

	if (!kernelFile.is_open())

	{

		std::cerr << "Failed to open file for reading: " << fileName << std::endl;

		return NULL;

	}

	std::ostringstream oss;

	oss << kernelFile.rdbuf();

	std::string srcStdStr = oss.str();

	const char *srcStr = srcStdStr.c_str();

	program = clCreateProgramWithSource(context, 1,

		(const char**)&srcStr,

		NULL, NULL);

	errNum = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);

	return program;

}
bool AssignmentProblemSolver::CreateMemObjects(cl_context context, cl_mem memObjects[6])

{
	// 创建输入内存对象
	memObjects[0] = clCreateBuffer(context,

		CL_MEM_READ_WRITE,  // 输入内存为只读，并可以从宿主机内存复制到设备内存

		nx* sizeof(int),		  // 输入内存空间大小

		NULL,

		NULL);

	memObjects[1] = clCreateBuffer(context,

		CL_MEM_READ_WRITE,  // 输入内存为只读，并可以从宿主机内存复制到设备内存

		ny * sizeof(int),		  // 输入内存空间大小

		NULL,

		NULL);

	memObjects[2] = clCreateBuffer(context,

		CL_MEM_READ_WRITE,  // 输入内存为只读，并可以从宿主机内存复制到设备内存

		nx * sizeof(double),		  // 输入内存空间大小

		NULL,

		NULL);

	memObjects[3] = clCreateBuffer(context,

		CL_MEM_READ_WRITE,  // 输入内存为只读，并可以从宿主机内存复制到设备内存

		1 * sizeof(double),		  // 输入内存空间大小

		NULL,

		NULL);

	memObjects[4] = clCreateBuffer(context,

		CL_MEM_READ_WRITE,  // 输入内存为只读，并可以从宿主机内存复制到设备内存

		nx * sizeof(double),		  // 输入内存空间大小

		NULL,

		NULL);

	memObjects[5] = clCreateBuffer(context,

		CL_MEM_READ_WRITE,  // 输入内存为只读，并可以从宿主机内存复制到设备内存

		ny * sizeof(double),		  // 输入内存空间大小

		NULL,

		NULL);

	if ((NULL == memObjects[0]) || (NULL == memObjects[1]) || (NULL == memObjects[2]) || NULL == memObjects[3] || NULL == memObjects[4] || NULL == memObjects[5])
	{
		cout << "Error creating memory objects" << endl;

		return false;
	}
	return true;
	
}
void Cleanup(cl_context context, cl_command_queue commandQueue,

	cl_program program, cl_kernel kernel, cl_mem memObjects[6])

{

	for (int i = 0; i < 6; i++)

	{

		if (memObjects[i] != 0)

			clReleaseMemObject(memObjects[i]);

	}

	if (commandQueue != 0)

		clReleaseCommandQueue(commandQueue);



	if (kernel != 0)

		clReleaseKernel(kernel);



	if (program != 0)

		clReleaseProgram(program);



	if (context != 0)

		clReleaseContext(context);
	return;
}
void AssignmentProblemSolver::GpuInitial()
{
	// 一、选择OpenCL平台并创建一个上下文

	context = CreateContext();

	// 二、 创建设备并创建命令队列

	commandQueue = CreateCommandQueue(context, &device);

	CreateMemObjects(context, memObjects);

	//三、创建和构建程序对象

	program = CreateProgram(context, device, "kernel.cl");

	kernel = clCreateKernel(program, "update", NULL);

	// 四、 创建OpenCL内核并分配内存空间
	

	errNum = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)(&nx));

	errNum |= clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)(&ny));

	errNum |= clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&memObjects[0]);

	errNum |= clSetKernelArg(kernel, 3, sizeof(cl_mem), (void *)&memObjects[1]);

	errNum |= clSetKernelArg(kernel, 4, sizeof(cl_mem), (void *)&memObjects[2]);

	errNum |= clSetKernelArg(kernel, 5, sizeof(cl_mem), (void *)&memObjects[3]);

	errNum |= clSetKernelArg(kernel, 6, sizeof(cl_mem), (void *)&memObjects[4]);

	errNum |= clSetKernelArg(kernel, 7, sizeof(cl_mem), (void *)&memObjects[5]);


	if (CL_SUCCESS != errNum)
	{
		cout << "Error setting kernel arguments" << endl;
	}
	// --------------------------10.运行内核---------------------------------

	globalWorkSize[0] = nx + ny;

}
/*
// --------------------------------------------------------------------------
// Usage example
// --------------------------------------------------------------------------
void main(void)
{
// Matrix size
int N=8; // tracks
int M=9; // detects
// Random numbers generator initialization
srand (time(NULL));
// Distance matrix N-th track to M-th detect.
vector< vector<double> > Cost(N,vector<double>(M));
// Fill matrix with random values
for(int i=0; i<N; i++)
{
for(int j=0; j<M; j++)
{
Cost[i][j] = (double)(rand()%1000)/1000.0;
std::cout << Cost[i][j] << "\t";
}
std::cout << std::endl;
}

AssignmentProblemSolver APS;

vector<int> Assignment;

cout << APS.Solve(Cost,Assignment) << endl;

// Output the result
for(int x=0; x<N; x++)
{
std::cout << x << ":" << Assignment[x] << "\t";
}

getchar();
}
*/
// --------------------------------------------------------------------------