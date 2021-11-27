#pragma once

#include <opencv2/opencv.hpp>
#include<vector>

using namespace cv;
using namespace std;

struct pointData {
    float response;
    Point point;
};

struct cornerResponse {
    bool operator()(pointData const &left, pointData const &right)
    {
        return left.response > right.response;
    }
};

struct keypoint_ {
    int x, y;
    vector<double> magnitude;
    vector<double> orientation;
    double scale;
    keypoint_(int x_, int y_, double scale_)
    {
        x = x_;
        y = y_;
        scale = scale_;
    }
};

struct Feature {
    int x, y;
    vector<double> features;
};