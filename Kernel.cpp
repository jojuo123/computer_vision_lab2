#include "Kernel.h"

cv::Mat_<double> Kernel::gauss_mat(int ker_size, double sigma_sq)
{
    cv::Mat_<double> kernel(ker_size, ker_size, 0.0);
    int half_size = (int)(ker_size / 2);
    double sum = 0.0;
    for (int i = 0; i < ker_size; i++)
        for (int j = 0; j < ker_size; j++)
        {
            int dist_i = abs(i - half_size), dist_j = abs(j - half_size);
            double p = ((double)dist_i * dist_i + (double)dist_j * dist_j) / (-2.0 * sigma_sq);
            // double g = 1.0 / (2.0 * PI * sigma_sq) * exp(p);
            double g = 1.0 / (2.0 * PI * sigma_sq) * exp(p);
            // double g = exp(p);
            kernel.at<double>(i, j) = g;
            sum += g;
        }
    for (int i = 0; i < ker_size; i++)
        for (int j = 0; j < ker_size; j++)
            kernel.at<double>(i, j) /= sum;
    return kernel;
}

cv::Mat_<double> Kernel::LoG(int ker_size, double sigma)
{
    cv::Mat_<double> kernel(ker_size, ker_size, 0.0);
    int half_size = (int)(ker_size / 2);
    double sum = 0.0;
    for (int i = 0; i < ker_size; i++)
        for (int j = 0; j < ker_size; j++)
        {
            int dist_i = abs(i - half_size), dist_j = abs(j - half_size);
            double p = ((double)dist_i * dist_i + (double)dist_j * dist_j) / (-2.0 * sigma * sigma);
            double g = -1.0 / (PI * sigma * sigma * sigma * sigma) * exp(p) * (1.0 + p);
            kernel.at<double>(i, j) = g;
            sum += 1.0 / (2.0 * PI * sigma * sigma) * exp(p);
        }
    // double scale = 1.0 / (PI * sigma * sigma * sigma * sigma);
    // for (int i = 0; i < ker_size; i++)
    //     for (int j = 0; j < ker_size; j++)
    //     {
    //         // kernel(i, j) /= scale;
    //         kernel(i, j) *= (sigma * sigma);
    //     }
    for (int i = 0; i < ker_size; i++)
        for (int j = 0; j < ker_size; j++)
            kernel.at<double>(i, j) /= sum;
    return kernel;
}

cv::Mat_<double> Kernel::normLoG(int ker_size, double sigma) //deprecated
{
    cv::Mat_<double> kernel(ker_size, ker_size, 0.0);
    // int half_size = (int)(ker_size / 2);
    // double sum = 0.0;
    // for (int i = 0; i < ker_size; i++)
    //     for (int j = 0; j < ker_size; j++)
    //     {
    //         int dist_i = abs(i - half_size), dist_j = abs(j - half_size);
    //         double p = ((double)dist_i * dist_i + (double)dist_j * dist_j) / (-2.0 * sigma * sigma);
    //         double g = -1.0 / (PI * sigma * sigma) * exp(p) * (1 + p);
    //         kernel.at<double>(i, j) = g;
    //         sum += g;
    //     }
    // for (int i = 0; i < ker_size; i++)
    //     for (int j = 0; j < ker_size; j++)
    //         kernel(i, j) = kernel(i,j) / sum;
    return kernel;
}

cv::Mat_<double> Kernel::avg_conv_mat(int ker_size)
{
    double value = 1.0 / ((double)ker_size * ker_size * 1.0);
    cv::Mat_<double> kernel(ker_size, ker_size, value);
    return kernel;
}

cv::Mat_<double> Kernel::SobelX()
{
    cv::Mat_<double> kernel(3, 3, 0.0);
    kernel.at<double>(0, 0) = 1.0;
    kernel.at<double>(0, 1) = 0.0;
    kernel.at<double>(0, 2) = -1.0;
    kernel.at<double>(1, 0) = 2.0;
    kernel.at<double>(1, 1) = 0.0;
    kernel.at<double>(1, 2) = -2.0;
    kernel.at<double>(2, 0) = 1.0;
    kernel.at<double>(2, 1) = 0.0;
    kernel.at<double>(2, 2) = -1.0;
    return kernel;
}

cv::Mat_<double> Kernel::SobelY()
{
    cv::Mat_<double> kernel(3, 3, 0.0);
    kernel.at<double>(0, 0) = 1.0;
    kernel.at<double>(0, 1) = 2.0;
    kernel.at<double>(0, 2) = 1.0;
    kernel.at<double>(1, 0) = 0.0;
    kernel.at<double>(1, 1) = 0.0;
    kernel.at<double>(1, 2) = 0.0;
    kernel.at<double>(2, 0) = -1.0;
    kernel.at<double>(2, 1) = -2.0;
    kernel.at<double>(2, 2) = -1.0;
    return kernel;
}

cv::Mat_<double> Kernel::Laplacian()
{
    cv::Mat_<double> k2(3, 3, 0.0);
    k2(0,1) = k2(1,0) = k2(1,2) = k2(2,1) = 1.0;
    k2(1, 1) = -4.0;
    return k2;
}