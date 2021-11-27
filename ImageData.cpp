#include "ImageData.h"

void ImageData::convolution_(Mat& result, Mat_<double>& kernel, int u, int v, int x, int y)
{
    int kernel_size = (int)(kernel.rows / 2);
    if (result.channels() == 3)
    {
        Vec3d conv(0.0, 0.0, 0.0);
        Vec3b& pixel_res = result.at<Vec3b>(x, y);
        for (int i = -kernel_size; i <= kernel_size; i++)
            for (int j = -kernel_size; j <= kernel_size; j++)
            {
                Vec3b& pixel = this->image.at<Vec3b>(u + i, v + j);
                int ker_pos_x = i + kernel_size, ker_pos_y = j + kernel_size;
                conv[0] += pixel[0] * kernel.at<double>(ker_pos_x, ker_pos_y);
                conv[1] += pixel[1] * kernel.at<double>(ker_pos_x, ker_pos_y);
                conv[2] += pixel[2] * kernel.at<double>(ker_pos_x, ker_pos_y);
            }
        pixel_res[0] = trunc((int)conv[0]);
        pixel_res[1] = trunc((int)conv[1]);
        pixel_res[2] = trunc((int)conv[2]);
    }
    else 
    {
        double conv = 0.0;
        uchar& pixel_res = result.at<uchar>(x, y);
        for (int i = -kernel_size; i <= kernel_size; i++)
            for (int j = -kernel_size; j <= kernel_size; j++)
            {
                uchar& pixel = this->image.at<uchar>(u + i, v + j);
                int ker_pos_x = i + kernel_size, ker_pos_y = j + kernel_size;
                conv += (double)pixel * kernel.at<double>(ker_pos_x, ker_pos_y);
            }
        pixel_res = trunc((int)conv);
    }
}

Mat ImageData::convolution(Mat_<double>& kernel)
{
    int ker_size = kernel.rows;
    int width = image.cols, height = image.rows;
    int h_out = height - (ker_size - 1), w_out = width - (ker_size - 1);

    Mat result;
    if (image.channels() == 3)
        result = Mat(Size(w_out, h_out), CV_8UC3);
    else
        result = Mat(Size(w_out, h_out), CV_8UC1);

    for (int i = 0; i < h_out; i++)
        for (int j = 0; j < w_out; j++)
        {
            int u = i + (int)(ker_size / 2);
            int v = j + (int)(ker_size / 2);
            convolution_(result, kernel, u, v, i, j);
        }
    return result;
}

int ImageData::colorContrast(int pixel, double f)
{
    return (int)(f * (pixel - (int)128) + (int)128);
}

int ImageData::trunc(int pixel)
{
    return min(max(0, pixel), 255);
}

Mat ImageData::changeBrightness(int g)
{
    int width = image.cols, height = image.rows;
    Mat result = Mat(Size(width, height), CV_8UC3);
    for (int y = 0; y < height; ++y)
        for (int x = 0; x < width; ++x)
        {
            if (image.channels() == 3)
            {
                Vec3b& pixel = image.at<Vec3b>(x, y);
                Vec3b& pixel_res = result.at<Vec3b>(x, y);
                pixel_res[0] = trunc(pixel[0] + g);
                pixel_res[1] = trunc(pixel[1] + g);
                pixel_res[2] = trunc(pixel[2] + g);
            }
        }
    return result;
}

Mat ImageData::changeContrast(int c)
{
    int width = image.cols, height = image.rows;
    double f = (259.0 * (c + 255.0)) / (255.0 * (259.0 - c));
    Mat result = Mat(Size(width, height), CV_8UC3);
    for (int y = 0; y < height; ++y)
        for (int x = 0; x < width; ++x)
        {
            if (image.channels() == 3)
            {
                Vec3b& pixel = image.at<Vec3b>(x, y);
                Vec3b& pixel_res = result.at<Vec3b>(x, y);
                pixel_res[0] = trunc(colorContrast(pixel[0], f));
                pixel_res[1] = trunc(colorContrast(pixel[1], f));
                pixel_res[2] = trunc(colorContrast(pixel[2], f));
            }
        }
    return result;
}



Mat ImageData::rbg2gray()
{
    int width = image.cols, height = image.rows;
    Mat result = Mat(Size(width, height), CV_8UC1);
    for (int y = 0; y < height; ++y)
        for (int x = 0; x < width; ++x)
        {
            if (image.channels() == 3)
            {
                Vec3b& pixel = image.at<Vec3b>(x, y);
                result.at<uchar>(x, y) = (uchar)((float)pixel[0] * 0.11 + (float)pixel[1] * 0.59 + (float)pixel[2] * 0.30);
            }
        }
    return result;
}

Mat& ImageData::getData()
{
    // TODO: insert return statement here
    return this->image;
}

ImageData::ImageData(string fname)
{
    image = imread(fname, IMREAD_COLOR);
    cvtColor(image, image_gray, COLOR_BGR2GRAY);
}

Mat ImageData::harrisDectect(int blockSize, double sigma, double k, double thresh, int apertureSize)
{

    Mat dst, dst_norm, dst_norm_scaled;
    dst = Mat(image.size(), CV_32FC1);

    cornerHarris(image_gray, dst, blockSize, apertureSize, k, BORDER_DEFAULT);
    normalize(dst, dst_norm, 0, 255, NORM_MINMAX, CV_32FC1, Mat());
    convertScaleAbs(dst_norm, dst_norm_scaled);
    Mat res = image;
    for (int i = 0; i < dst_norm.rows; i++)
        for (int j = 0; j < dst_norm.cols; j++)
        {
            if ((int)dst_norm.at<float>(i, j) > (int)thresh)
            {
                circle(res, Point(j, i), 5, Scalar(0), 2, 8, 0);
            }
        }
    return res;

    // ImageData gray = ImageData(image_gray);
    // Mat_<double> gauss_kernel = Kernel::gauss_mat(3, sigma * sigma);
    // Mat tmp = gray.convolution(gauss_kernel);
    /*
    DoubleMat dMat = DoubleMat(image_gray);
    Mat_<double> sobelx = Kernel::SobelX();
    Mat_<double> sobely = Kernel::SobelY();
    Mat_<double> ix = dMat.convolution(sobelx);
    Mat_<double> iy = dMat.convolution(sobely);
    // Mat_<double> ixy = DoubleMat::times(ix, iy);

    Mat_<double> Ix, Iy, Ixy;
    copyMakeBorder(ix, Ix, 1, 1, 1, 1, BORDER_CONSTANT, 0.0);
    copyMakeBorder(iy, Iy, 1, 1, 1, 1, BORDER_CONSTANT, 0.0);
    Ixy = DoubleMat::times(Ix, Iy);
    Mat_<double> Ixx = DoubleMat::times(Ix, Ix);
    Mat_<double> Iyy = DoubleMat::times(Iy, Iy);

    int halfsize = (int)(blockSize / 2);

    Mat_<double> gauss_dev = Kernel::gauss_mat(blockSize, sigma*sigma);
    Mat_<double> Ixx_gauss = DoubleMat(Ixx).convolution(gauss_dev);
    Mat_<double> Iyy_gauss = DoubleMat(Iyy).convolution(gauss_dev);
    Mat_<double> Ixy_gauss = DoubleMat(Ixy).convolution(gauss_dev);

    Mat_<int> response = DoubleMat::HarrisResponse(Ixx_gauss, Iyy_gauss, Ixy_gauss, k, thresh);
    Mat_<int> response_pad;
    copyMakeBorder(response, response_pad, halfsize, halfsize, halfsize, halfsize, BORDER_CONSTANT, 0);

    Mat_<int> nonMax = DoubleMat::nonmaxSus(response_pad, blockSize);

    // cout << response_pad.rows;

    Mat res = this->image;
    for (int x = 0; x < nonMax.cols; ++x)
    {
        for (int y = 0; y < nonMax.rows; ++y)
        {
            if (nonMax(x, y) > 0)
            {
                circle(res, Point(x, y), 5, Scalar(0), 2, 8, 0);
            }
            // cout << nonMax(x, y) << " ";
        }
        // cout << endl;
    }

    return res;*/
}

Mat ImageData::blobDetect(int ker_size, double threshhold)
{
    int dx[] = {1, 1, 1, 0, 0, 0, -1, -1, -1};
    int dy[] = {1, 1, -1, 1, 0, -1, 1, 0, -1};
    // Mat res;
    // Ptr<Feature2D> sbd = SimpleBlobDetector::create();
    // vector<KeyPoint> keypoints;
    // sbd->detect(image, keypoints);
    // drawKeypoints(image, keypoints, res);
    // for (vector<KeyPoint>::iterator k = keypoints.begin(); k != keypoints.end(); ++k)
    //     circle(res, k->pt, (int)k->size, Scalar(0, 0, 255), 2);
    // return res;
    Mat bordered_gray;
    int bor_size = (int)(ker_size / 2);
    copyMakeBorder(image_gray, bordered_gray, bor_size, bor_size, bor_size, bor_size, BORDER_CONSTANT, 0);
    DoubleMat lap = DoubleMat(bordered_gray);
    // lap.norm();
    vector<Mat_<double> > scaleSpace;
    vector<double> scales;

    int iter = 1;
    while (iter <= 30)
    {
        double sigma = (double)iter;
        // Mat_<double> kernel = Kernel::gauss_mat(ker_size, sigma);
        Mat_<double> kernel = Kernel::LoG(ker_size, sigma);
        Mat_<double> response = lap.convolution(kernel);
        for (int i = 0; i < response.rows; i++)
        {
            for (int j = 0; j < response.cols; j++)
            {
                double tmp = response(i, j);
                response(i, j) = tmp * tmp * sigma * sigma * sigma * sigma;
            }
        }

        response = DoubleMat::normalize(response);
        scaleSpace.push_back(response);
        scales.push_back(sigma);
        iter++;
    }
    Mat res = image;
    int width = image.cols;
    int height = image.rows;

    for (int x = 1; x < width-1; x++)
    {
        for (int y = 1; y < height-1; y++)
        {
            bool maxFound = true;
            double maxScale = 0.0;
            double maxResponse = 0.0;
            int tmp_i = -1;
            for (int i = 1; i < (scales.size()-1); i++)
            {
                maxFound = true;
                double pivot = scaleSpace[i](y, x);
                for (int k = 0; k < 9; k++)
                {
                    for (int z = -1; z <= 1; z++)
                    {
                        if (z == 0 && dx[k] == 0 && dy[k] == 0)
                            continue;
                        int u = x + dx[k], v = y + dy[k], t = i + z;
                        if (t >= scaleSpace.size() || t < 0)
                            continue;
                        // cout << u << " " << v << " " << t << endl;
                        if (pivot <= scaleSpace[t](v, u))
                        {
                            maxFound = false;
                            break;
                        }
                    }
                    if (!maxFound)
                        break;
                }
                if (maxFound && maxResponse < pivot)
                {
                    maxScale = scales[i];
                    maxResponse = pivot;
                }
            }

            if (maxResponse > threshhold)
            {
                // cout << maxScale << " ";
                int rad = (int)(maxScale * sqrt(2));
                if (rad == 0) rad = 1;
                // f << x << " " << y << " " << rad << " " << maxResponse << endl;
                circle(res, Point(x, y), rad, Scalar(0, 0, 255), 1);
            }
        }
    }
    // f.close();
    return res;
}

ImageData::ImageData(Mat image)
{
    this->image = image;
    if (image.channels() == 3)
        cvtColor(image, image_gray, COLOR_BGR2GRAY);
    else 
        image_gray = this->image;
}

vector<KeyPoint> ImageData::harrisKeypoints(int blockSize, double sigma, double k, int thresh, int apertureSize)
{
    Mat dst, dst_norm, dst_norm_scaled;
    dst = Mat(image.size(), CV_32FC1);
    float size = (float)sigma * sqrt(2);

    cornerHarris(image_gray, dst, blockSize, apertureSize, k, BORDER_DEFAULT);
    normalize(dst, dst_norm, 0, 255, NORM_MINMAX, CV_32FC1, Mat());
    convertScaleAbs(dst_norm, dst_norm_scaled);
    vector<KeyPoint> keys; 
    for (int i = 0; i < dst_norm.rows; i++)
        for (int j = 0; j < dst_norm.cols; j++)
        {
            if ((int)dst_norm.at<float>(i, j) > thresh)
            {
                keys.push_back(KeyPoint(i, j, size));
            }
        }
    return keys;
}

Mat ImageData::Sift(vector<KeyPoint>& kp, bool harris, int blockSize, double sigma, int thresh, double k, int apertureSize)
{
    // Mat dst, dst_norm, dst_norm_scaled;
    // dst = Mat(image.size(), CV_32FC1);
    // float size = (float)sigma * sqrt(2);

    // cornerHarris(image_gray, dst, blockSize, apertureSize, k, BORDER_DEFAULT);
    // normalize(dst, dst_norm, 0, 255, NORM_MINMAX, CV_32FC1, Mat());
    // convertScaleAbs(dst_norm, dst_norm_scaled);
    vector<KeyPoint> keys = harrisKeypoints(blockSize, sigma, k, thresh, apertureSize);
    // for (int i = 0; i < dst_norm.rows; i++)
    //     for (int j = 0; j < dst_norm.cols; j++)
    //     {
    //         if ((int)dst_norm.at<float>(i, j) > thresh)
    //         {
    //             keys.push_back(KeyPoint(j, i, size));
    //         }
    //     }
    kp = keys;
    Mat descriptors;
    Ptr<SIFT> siftPtr = SIFT::create();
    siftPtr->compute(image_gray, keys, descriptors);
    return descriptors;
}

Mat ImageData::SiftBlob(vector<KeyPoint>& kp, int ker_size, double threshhold)
{
    vector<KeyPoint> keys = blobKeyPoints(ker_size, threshhold);
    kp = keys;
    Mat descriptors;
    Ptr<SIFT> siftPtr = SIFT::create();
    siftPtr->compute(image_gray, keys, descriptors);
    return descriptors;
}

Mat ImageData::DoGSift(vector<KeyPoint>& kp)
{
    kp = DoG();
    Mat descriptors;
    Ptr<SIFT> siftPtr = SIFT::create();
    siftPtr->compute(image_gray, kp, descriptors);
    return descriptors;
}

vector<KeyPoint> ImageData::DoG()
{
    vector<KeyPoint> res;
    Ptr<SIFT> siftPtr = SIFT::create();
    siftPtr->detect(image_gray, res);
    return res;
}

Mat ImageData::DoGDetect()
{
    vector<KeyPoint> kp = DoG();
    Mat res;
    drawKeypoints(image, kp, res);
    return res;
}

vector<KeyPoint> ImageData::blobKeyPoints(int ker_size, double threshhold)
{
    vector<KeyPoint> res;
    int dx[] = {1, 1, 1, 0, 0, 0, -1, -1, -1};
    int dy[] = {1, 1, -1, 1, 0, -1, 1, 0, -1};
    // Mat res;
    // Ptr<Feature2D> sbd = SimpleBlobDetector::create();
    // vector<KeyPoint> keypoints;
    // sbd->detect(image, keypoints);
    // drawKeypoints(image, keypoints, res);
    // for (vector<KeyPoint>::iterator k = keypoints.begin(); k != keypoints.end(); ++k)
    //     circle(res, k->pt, (int)k->size, Scalar(0, 0, 255), 2);
    // return res;
    Mat bordered_gray;
    int bor_size = (int)(ker_size / 2);
    copyMakeBorder(image_gray, bordered_gray, bor_size, bor_size, bor_size, bor_size, BORDER_CONSTANT, 0);
    DoubleMat lap = DoubleMat(bordered_gray);
    // lap.norm();
    vector<Mat_<double> > scaleSpace;
    vector<double> scales;

    int iter = 1;
    while (iter <= 30)
    {
        double sigma = (double)iter;
        // Mat_<double> kernel = Kernel::gauss_mat(ker_size, sigma);
        Mat_<double> kernel = Kernel::LoG(ker_size, sigma);
        Mat_<double> response = lap.convolution(kernel);
        for (int i = 0; i < response.rows; i++)
        {
            for (int j = 0; j < response.cols; j++)
            {
                double tmp = response(i, j);
                response(i, j) = tmp * tmp * sigma * sigma * sigma * sigma;
            }
        }

        response = DoubleMat::normalize(response);
        scaleSpace.push_back(response);
        scales.push_back(sigma);
        iter++;
    }
    int width = image.cols;
    int height = image.rows;

    for (int x = 1; x < width-1; x++)
    {
        for (int y = 1; y < height-1; y++)
        {
            bool maxFound = true;
            double maxScale = 0.0;
            double maxResponse = 0.0;
            int tmp_i = -1;
            for (int i = 1; i < (scales.size()-1); i++)
            {
                maxFound = true;
                double pivot = scaleSpace[i](y, x);
                for (int k = 0; k < 9; k++)
                {
                    for (int z = -1; z <= 1; z++)
                    {
                        if (z == 0 && dx[k] == 0 && dy[k] == 0)
                            continue;
                        int u = x + dx[k], v = y + dy[k], t = i + z;
                        if (t >= scaleSpace.size() || t < 0)
                            continue;
                        // cout << u << " " << v << " " << t << endl;
                        if (pivot <= scaleSpace[t](v, u))
                        {
                            maxFound = false;
                            break;
                        }
                    }
                    if (!maxFound)
                        break;
                }
                if (maxFound && maxResponse < pivot)
                {
                    maxScale = scales[i];
                    maxResponse = pivot;
                }
            }

            if (maxResponse > threshhold)
            {
                // cout << maxScale << " ";
                int rad = (int)(maxScale * sqrt(2));
                if (rad == 0) rad = 1;
                // f << x << " " << y << " " << rad << " " << maxResponse << endl;
                // circle(res, Point(x, y), rad, Scalar(0, 0, 255), 1);
                res.push_back(KeyPoint(x, y, rad*2));
            }
        }
    }
    // f.close();
    return res;
}

Mat ImageData::lbp(vector<KeyPoint>& kp, int blockSize, double sigma, int thresh, double k, int apertureSize, int gridx, int gridy)
{
    vector<KeyPoint> keys = harrisKeypoints(blockSize, sigma, k, thresh, apertureSize);
    kp = keys;
    Mat des = LBP::getFeat(image_gray, keys, gridx, gridy);
    return des;
}

Mat ImageData::lbpBlob(vector<KeyPoint>& kp, int ker_size, double threshhold, int gridx, int gridy)
{
    vector<KeyPoint> keys = blobKeyPoints(ker_size, threshhold);
    kp = keys;
    Mat descriptors = LBP::getFeat(image_gray, keys, gridx, gridy);
    return descriptors;
}

Mat ImageData::lbpDoG(vector<KeyPoint>& kp, int gridx, int gridy)
{
    kp = DoG();
    Mat descriptors = LBP::getFeat(image_gray, kp, gridx, gridy);
    return descriptors;
}

Mat ImageData::matches(Mat& im1, Mat& im2, vector<KeyPoint> kp1, vector<KeyPoint> kp2, Mat& des1, Mat& des2, Mat& dst, int k)
{
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);
    vector<vector<DMatch> > knn_matches;
    matcher->knnMatch(des1, des2, knn_matches, k);
    drawMatches( im1, kp1, im2, kp2, knn_matches, dst);
    return dst;
}
