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
    // cout << dst.rows << " " << dst.cols << endl;
    Mat res(kp.size(), 256, CV_32FC1);
    int hc = (int)(gridx / 2);
    int hr = (int)(gridy / 2);
    int nf = 0;
    for (KeyPoint k : kp)
    {
        int pr = (int)k.pt.y;
        int pc = (int)k.pt.x;
        int sr = max(0, pr-hr);
        int sc = max(0, pc-hc);
        int fr, fc;
        if (sr + gridy > src.rows-1)
        {
            fr = src.rows-1;
            sr = max(0, fr-gridy);
        }
        else 
        {
            fr = sr+gridy;
        }
        if (sc + gridx > src.cols-1)
        {
            fc = src.cols-1;
            sc = max(0, fc-gridx);
        }
        else
        {
            fc = sc+gridx;
        }
        // cout << px << " " << py << endl;
        // cout << sr << " " << sc << endl << fr <<  " " << fc << endl << endl;

        Mat cell = Mat(dst, Rect(sc, sr, fc-sc, fr-sr));
        Mat hist = histogram(cell, 256);
        Mat hist_;
        normalize(hist, hist_, 0, 255, NORM_MINMAX, CV_32SC1);
        for (int i = 0; i < 256; i++)
            res.at<float>(nf, i) = (float)hist_.at<int>(0, i);
        nf++;
    }
    // cout << res.rows <<  " " << res.cols << endl;
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