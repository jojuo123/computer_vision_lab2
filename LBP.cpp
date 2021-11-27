#include "LBP.h"

void LBP::create(const Mat& src, Mat& dst)
{
    int dx[] = {0, 1, 1, 1, 0, -1, -1, -1};
    int dy[] = {-1, -1, 0, 1, 1, 1, 0, -1};
    dst = Mat::zeros(src.rows, src.cols, CV_8UC1);
    for (int i = 1; i < src.rows-1; i++)
        for (int j = 1; j < src.cols-1; j++)
        {
            int center = (int)src.at<uchar>(i, j); 
            unsigned char code = 0;
            for (int k = 0; k < 8; k++)
            {
                int u = i + dx[k], v = j + dy[k];
                int p = (int)src.at<uchar>(u, v);
                code |= (center < p) << k;
            }
            dst.at<unsigned char>(i, j) = code;
        }
}

// vector<Mat> LBP::spatical_hist(const Mat& src, int numPatterns, int gridx, int gridy, int overlap)
// {
//     int w = static_cast<int>(floor(src.cols/gridx));
//     int h = static_cast<int>(floor(src.rows/gridy));
//     int width = src.cols;
//     int height = src.rows;
//     vector<Mat> histograms;
//     for (int x=0; x < width - w; x+=(w-overlap))
//     {
//         for (int y=0; x < height - h; y+=(h-overlap))
//         {
//             Mat cell = Mat(src, Rect(x, y, w, h));
//             histograms.push_back(histogram(cell, numPatterns));
//         }
//     }
// }

Mat LBP::getFeat(const Mat& src, vector<KeyPoint>& kp, int gridx, int gridy)
{
    Mat dst;
    LBP::create(src, dst);
    Mat res(kp.size(), 256, CV_32SC1);
    int hx = (int)(gridx / 2);
    int hy = (int)(gridy / 2);
    int nf = 0;
    for (KeyPoint k : kp)
    {
        int px = (int)k.pt.y;
        int py = (int)k.pt.x;
        int sx = max(0, px-hx);
        int sy = max(0, py-hy);
        int fx = min(src.cols, sx+gridx);
        int fy = min(src.rows, sy+gridy);
        // cout << px << " " << py << endl;
        // cout << sx << " " << fx << " " << sy <<  " " << fy << endl;
        Mat cell = Mat(dst, Rect(sx, sy, fx-sx, fy-sy));
        Mat hist = histogram(cell, 256);
        Mat hist_;
        normalize(hist, hist_, 0, 255, NORM_MINMAX, CV_32SC1);
        for (int i = 0; i < 256; i++)
            res.at<int>(nf, i) = hist_.at<int>(0, i);
        nf++;
    }
    return res;
}

Mat LBP::histogram(const Mat& src, int numPattern)
{
    Mat hist = Mat::zeros(1, numPattern, CV_32SC1);
    for (int i = 0; i < src.rows; i++)
        for (int j = 0; j < src.cols; j++)
        {
            int bin = src.at<unsigned char>(i, j);
            hist.at<int>(0, bin) += 1;
        }
    return hist;
}