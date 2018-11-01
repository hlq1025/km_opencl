#include <CL/cl.h>

#include <iostream>

#include <fstream>

#include <stdio.h>

#include <vector>

#include <string>

#include<algorithm>

#include <opencv2/core/core.hpp>

#include <opencv2/highgui/highgui.hpp>

#include"km.h"
using namespace std;

using namespace cv;

int main()
{
	
	int N = 49; // tracks
	int M = 50; // detects
	// Random numbers generator initialization
	
	// Distance matrix N-th track to M-th detect.
	double**w;
	w = new double*[N];
	for (int i = 0; i < N; i++)
	{
		w[i] = new double[M];
	}
	// Fill matrix with random values
	for (int i = 0; i<N; i++)
	{
		for (int j = 0; j<M; j++)
		{
			w[i][j] = (double)(rand() % 1000) / 1000.0;
			std::cout << w[i][j] << "\t";
		}
		std::cout << std::endl;
	}
	AssignmentProblemSolver APS(N,M,w);

	cout << APS.solve() << endl;
}
