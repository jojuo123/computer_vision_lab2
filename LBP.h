#pragma one

#include <math.h>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>

using namespace cv;
using namespace std;

class LBP
{
public:
    static void create(const Mat& src, Mat& dst);
    static Mat getFeat(const Mat& src, vector<KeyPoint>& kp, int gridx, int gridy);
    static Mat histogram(const Mat& src, int numPattern);
};