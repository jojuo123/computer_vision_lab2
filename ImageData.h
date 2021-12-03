#pragma once

#define _USE_MATH_DEFINES
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/core.hpp>
#include <string.h>
#include <math.h>
#include <vector>
#include "DoubleMat.h"
#include "LBP.h"
using namespace cv;
using namespace std;


class ImageData
{
private:
	Mat image;
	Mat image_gray;
	void convolution_(Mat& result, Mat_<double>& kernel, int u, int v, int x, int y);
	int colorContrast(int pixel, double f);
	int trunc(int pixel);
	int circleIntersect(int x1, int y1, int x2,
           int y2, int r1, int r2)
	{
		int distSq = (x1 - x2) * (x1 - x2) +
					(y1 - y2) * (y1 - y2);
		int radSumSq = (r1 + r2) * (r1 + r2);
		if (distSq == radSumSq)
			return 1;
		else if (distSq > radSumSq)
			return -1;
		else
			return 0;
	}
public:
	Mat changeBrightness(int g);
	Mat changeContrast(int c);
	Mat convolution(Mat_<double>& kernel);
	Mat rbg2gray();
	Mat& getData();
	ImageData(){}
	ImageData(string fname);
	ImageData(Mat image);

	vector<KeyPoint> harrisKeypoints(int blockSize, double sigma, double k, int thresh, int apertureSize=3);
	Mat harrisDectect(int blockSize, double sigma, double k, int thresh, int apertureSize);
	Mat blobDetect(int ker_size=3, double threshhold=0.5);
	vector<KeyPoint> blobKeyPoints(int ker_size=3, double threshhold=0.5);
	Mat Sift(vector<KeyPoint>& kp, bool harris=true, int blockSize=3, double sigma=1.0, int thresh=0, double k=0.04, int apertureSize=3);
	Mat SiftBlob(vector<KeyPoint>& kp, int ker_size, double threshhold);
	vector<KeyPoint> DoG();
	Mat DoGDetect();
	Mat DoGSift(vector<KeyPoint>& kp);
	Mat lbp(vector<KeyPoint>& kp, int blockSize=3, double sigma=1.0, int thresh=165, double k=0.04, int apertureSize=3, int gridx=16, int gridy=16);
	Mat lbpBlob(vector<KeyPoint>& kp, int ker_size=3, double threshhold=0.5, int gridx=16, int gridy=16);
	Mat lbpDoG(vector<KeyPoint>& kp, int gridx=16, int gridy=16);
	static Mat matches(Mat& im1, Mat& im2, vector<KeyPoint> kp1, vector<KeyPoint> kp2, Mat& des1, Mat& des2, Mat& dst, int k = 2);

};