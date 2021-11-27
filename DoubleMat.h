#pragma once

#include <opencv2/opencv.hpp>
#include "Kernel.h"
#include "structures.h"
#include <opencv2/features2d.hpp>

using namespace std;
using namespace cv;

class DoubleMat
{
private:
    Mat_<double> mat;
public:
    void convolution_(Mat_<double>& result, Mat_<double>& kernel, int u, int v, int x, int y);
    Mat_<double> convolution(Mat_<double>& kernel);
    DoubleMat(Mat im);
    DoubleMat(Mat_<double> im);
    static Mat_<double> times(Mat_<double> const &a, Mat_<double> const &b);
    static Mat_<int> HarrisResponse(Mat_<double> &Ixx, Mat_<double> &Iyy, Mat_<double> &Ixy, double k, double threshhold);
    static Mat_<int> nonmaxSus(Mat_<int> response, int blockSize);
    static Mat_<double> squaredMat(Mat_<double> m);
    static Mat_<double> scaleMul(Mat_<double> m, double scale);
    static Mat_<double> normalize(Mat_<double> m);
    void norm();
};