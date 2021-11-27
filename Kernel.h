#pragma once
#define _USE_MATH_DEFINES
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string.h>
#include <fstream>
#include <math.h>

const double PI = M_PI;

class Kernel
{
public:
	static cv::Mat_<double> gauss_mat(int ker_size, double sigma_sq = 4.0);
	static cv::Mat_<double> avg_conv_mat(int ker_size);
	static cv::Mat_<double> SobelX();
	static cv::Mat_<double> SobelY();
	static cv::Mat_<double> LoG(int ker_size, double sigma);
	static cv::Mat_<double> normLoG(int ker_size, double sigma);
	static cv::Mat_<double> Laplacian();
};