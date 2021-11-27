#include "DoubleMat.h"

void DoubleMat::convolution_(Mat_<double>& result, Mat_<double>& kernel, int u, int v, int x, int y)
{
    int kernel_size = (int)(kernel.rows / 2);
    double conv = 0.0;
    double& pixel_res = result.at<double>(x, y);
    for (int i = -kernel_size; i <= kernel_size; i++)
        for (int j = -kernel_size; j <= kernel_size; j++)
        {
            double& pixel = this->mat.at<double>(u + i, v + j);
            int ker_pos_x = i + kernel_size, ker_pos_y = j + kernel_size;
            conv += (double)pixel * kernel.at<double>(ker_pos_x, ker_pos_y);
        }
    pixel_res = conv;
}

Mat_<double> DoubleMat::convolution(Mat_<double>& kernel)
{
    int ker_size = kernel.rows;
    int width = mat.cols, height = mat.rows;
    int h_out = height - (ker_size - 1), w_out = width - (ker_size - 1);

    // cout << h_out << "and" << w_out << endl;

    Mat_<double> result = Mat_<double>(h_out, w_out, 0.0);
    // cout << result.at<double>(264, 191) << endl;


    for (int i = 0; i < h_out; i++)
        for (int j = 0; j < w_out; j++)
        {
            int u = i + (int)(ker_size / 2);
            int v = j + (int)(ker_size / 2);
            //u->x, v->y
            convolution_(result, kernel, u, v, i, j);
        }
    return result;
}

DoubleMat::DoubleMat(Mat im)
{
    mat = Mat_<double>(im);
}

DoubleMat::DoubleMat(Mat_<double> im)
{
    mat = im;
}

Mat_<double> DoubleMat::times(Mat_<double> const &a, Mat_<double> const &b)
{
    int width = a.cols;
    int height = a.rows;
    Mat_<double> res(height, width, 0.0);
    for (int i = 0; i < width; ++i)
    {
        for (int j = 0; j < height; ++j)
        {
            res(i, j) = a(i, j) * b(i, j);
        }
    }
    return res;
}

Mat_<int> DoubleMat::HarrisResponse(Mat_<double> &Ixx, Mat_<double> &Iyy, Mat_<double> &Ixy, double k, double threshhold)
{
    int width = Ixx.cols;
    int height = Ixx.rows;

    Mat_<int> R = Mat_<int>(height, width, 0);

    int maxVal = 0;
    for (int x = 0; x < width; ++x)
    {
        for (int y = 0; y < height; ++y)
        {
            double r = Ixx(x, y) * Iyy(x, y) - Ixy(x, y) * Ixy(x, y) - k * (Ixx(x, y) + Iyy(x, y)) * (Ixx(x, y) + Iyy(x, y));
            // cout << r <<  " ";
            R(x, y) = (int)r;
            if (r > maxVal) 
                maxVal = r;
            // if (R(x, y) < threshhold)
            //     R(x, y) = 0;
        }
    }

    int t = (int)(threshhold * maxVal);
    for (int x = 0; x < width; ++x)
    {
        for (int y = 0; y < height; ++y)
        {
            if (R(x, y) < t)
                R(x, y) = 0;
            // cout << R(x, y) << " ";
        }
    }

    return R;
}

Mat_<int> DoubleMat::nonmaxSus(Mat_<int> response, int blockSize)
{
    int width = response.cols;
    int height = response.rows;
    int halfsize = (int)(blockSize / 2);
    int sx = halfsize, fx = width - fx, sy = halfsize, fy = height - halfsize;

    for (int x = sx; x < fx; ++x)
    {
        for (int y = sy; y < fy; ++y)
        {
            int maxVal = 0;
            for (int i = -halfsize; i <= halfsize; ++i)
                for (int j = -halfsize; j <= halfsize; ++j)
                    if (response(x+i, y+j) > maxVal)
                        maxVal = response(x+i, y+j);
            for (int i = -halfsize; i <= halfsize; ++i)
                for (int j = -halfsize; j <= halfsize; ++j)
                    if (response(x+i, x+j) < maxVal)
                        response(x+i, x+j) = 0;
        }
    }

    return response;
}

Mat_<double> DoubleMat::squaredMat(Mat_<double> m)
{
    Mat_<double> res = Mat_<double>(m.rows, m.cols, 0.0);
    for (int x = 0; x < m.cols; x++)
        for (int y = 0; y < m.rows; y++)
            res(x, y) = m(x, y) * m(x, y);
    return res;
        
}

Mat_<double> DoubleMat::scaleMul(Mat_<double> m, double scale)
{
    Mat_<double> res = Mat_<double>(m.rows, m.cols, 0.0);
    for (int x = 0; x < m.cols; x++)
        for (int y = 0; y < m.rows; y++)
            res(x, y) = m(x, y) * scale;
    return res;
}

Mat_<double> DoubleMat::normalize(Mat_<double> m)
{
    double vmin = m(0, 0), vmax = m(0, 0);
    for (int i = 0; i < m.rows; ++i)
    {
        for (int j = 0; j < m.cols; ++j)
        {
            double val = m(i,j);
            vmin = min(vmin, val);
            vmax = max(vmax, val);
        }
    }

    double denom = vmax - vmin;
    for (int i = 0; i < m.rows; ++i)
    {
        for (int j = 0; j < m.cols; ++j)
        {
            double val = m(i, j);
            m(i,j) = (val - vmin) / denom;
        }
    }
    return m;
}

void DoubleMat::norm()
{
    mat = DoubleMat::normalize(mat);
}