#include "ImageData.h"
#include <chrono>

using namespace cv;
using namespace std;
const string arg_const[] = {"harris", "blob", "dog", "m", "sift", "lbp"};
const int FPS = 30;
const int MAX_W = 640;
const int MAX_H = 440;
bool isCam = false;

int ker_size_tb = 0, sigma_tb = 0, threshhold_tb = 0, aperture_size_tb = 0, harris_k_param = 0;
int gridx_tb = 0, gridy_tb = 0;
int knn = 0;

const int MAX_KER_SIZE = 19;
const int MAX_SIGMA = 15;
const int MAX_K = 10;
const int MAX_THRESH = 200;
const int MAX_APERTURE = 5;
const int MAX_THRESH_BLOB = 10;
const int MAX_GRID_SIZE = 32;
const int MAX_KNN = 10;

void TrackbarCallbackFunction(int, void*)
{
    // int valueFromUser = *(static_cast<int*>(userData));
}

bool handleImage(char** argv, Mat image, bool isCam)
{
    ImageData im;
    if (isCam)
        im = ImageData(image);
    // Mat res = im.blobDetect(19, 0.4);
    // imshow("show image", res);
    // return;

    int ker_size = max((ker_size_tb % 2 == 0) ? ker_size_tb + 1 : ker_size_tb, 3);
    double sigma = (sigma_tb == 0) ? 1.0 : (double)sigma_tb;
    int threshhold = max(100, threshhold_tb);
    int aperture_size = max(3, (aperture_size_tb % 2 == 0) ? aperture_size_tb + 1 : aperture_size_tb);
    double harris_k = 0.04 + (((double)harris_k_param) * 0.02) / (MAX_K);
    double thresh = ((double)threshhold_tb) / (MAX_THRESH_BLOB);
    int gridx = max(1, gridx_tb);
    int gridy = max(1, gridy_tb);
    int knn_val = max(knn, 2);
    
    if (arg_const[3].compare(argv[1]) != 0)
    {
        if (arg_const[0].compare(argv[1]) == 0)
        {
            if (!isCam)
                im = ImageData(argv[2]);
            // int ker_size = (ker_size_tb % 2 == 0) ? ker_size_tb + 1 : ker_size_tb;
            // double sigma = (sigma_tb == 0) ? 1.0 : (double)sigma_tb;
            // int threshhold = threshhold_tb;
            // int aperture_size = max(1, aperture_size_tb);
            // double harris_k = 0.04 + (((double)harris_k_param) * 0.02) / (MAX_K);
            Mat res = im.harrisDectect(ker_size, sigma, harris_k, threshhold, aperture_size);
            imshow("show image", res);
            return true;
        }
        else if (arg_const[1].compare(argv[1]) == 0)
        {
            if (!isCam)
                im = ImageData(argv[2]);
            // int ker_size = (ker_size_tb % 2 == 0) ? ker_size_tb + 1 : ker_size_tb;
            // double thresh = ((double)threshhold_tb) / (MAX_THRESH_BLOB);
            Mat res = im.blobDetect(ker_size, thresh);
            imshow("show image", res);
            return true;
        }
        else if (arg_const[2].compare(argv[1]) == 0)
        {
            if (!isCam)
                im = ImageData(argv[2]);
            Mat res = im.DoGDetect();
            imshow("show image", res);
            return true;
        }
    }
    else 
    {
        ImageData im1, im2;
        Mat dst;
        if (arg_const[4].compare(argv[3]) == 0)
        {
            if (arg_const[0].compare(argv[2]) == 0)
            {
                // int ker_size = (ker_size_tb % 2 == 0) ? ker_size_tb + 1 : ker_size_tb;
                // double sigma = (sigma_tb == 0) ? 1.0 : (double)sigma_tb;
                // int threshhold = threshhold_tb;
                // int aperture_size = max(1, aperture_size_tb);
                // double harris_k = 0.04 + (((double)harris_k_param) * 0.02) / (MAX_K);
                im1 = ImageData(argv[4]);
                im2 = ImageData(argv[5]);
                vector<KeyPoint> kp1, kp2;
                Mat des1 = im1.Sift(kp1, true, ker_size, sigma, threshhold, harris_k, aperture_size);
                Mat des2 = im2.Sift(kp2, true, ker_size, sigma, threshhold, harris_k, aperture_size);
                Mat res = ImageData::matches(im1.getData(), im2.getData(), kp1, kp2, des1, des2, dst, knn_val);
                imshow("show image", res);
                return true;
            }
            else if (arg_const[1].compare(argv[2]) == 0)
            {
                // int ker_size = (ker_size_tb % 2 == 0) ? ker_size_tb + 1 : ker_size_tb;
                // double thresh = ((double)threshhold_tb) / (MAX_THRESH_BLOB);
                im1 = ImageData(argv[4]);
                im2 = ImageData(argv[5]);
                vector<KeyPoint> kp1, kp2;
                Mat des1 = im1.SiftBlob(kp1, ker_size, thresh);
                Mat des2 = im2.SiftBlob(kp2, ker_size, thresh);
                Mat res = ImageData::matches(im1.getData(), im2.getData(), kp1, kp2, des1, des2, dst, knn_val);
                imshow("show image", res);
                return true;
            }
            else if (arg_const[2].compare(argv[2]) == 0)
            {
                im1 = ImageData(argv[4]);
                im2 = ImageData(argv[5]);
                vector<KeyPoint> kp1, kp2;
                Mat des1 = im1.DoGSift(kp1);
                Mat des2 = im2.DoGSift(kp2);
                Mat res = ImageData::matches(im1.getData(), im2.getData(), kp1, kp2, des1, des2, dst, knn_val);
                imshow("show image", res);
                return true;
            }
        }
        else if (arg_const[5].compare(argv[3]) == 0)
        {
            if (arg_const[0].compare(argv[2]) == 0)
            {
                // int ker_size = (ker_size_tb % 2 == 0) ? ker_size_tb + 1 : ker_size_tb;
                // double sigma = (sigma_tb == 0) ? 1.0 : (double)sigma_tb;
                // int threshhold = threshhold_tb;
                // int aperture_size = max(1, aperture_size_tb);
                // double harris_k = 0.04 + (((double)harris_k_param) * 0.02) / (MAX_K);
                // int gridx = max(1, gridx_tb);
                // int gridy = max(1, gridy_tb);
                im1 = ImageData(argv[4]);
                im2 = ImageData(argv[5]);
                vector<KeyPoint> kp1, kp2;
                Mat des1 = im1.lbp(kp1, ker_size, sigma, threshhold, harris_k, aperture_size, gridx, gridy);
                Mat des2 = im2.lbp(kp2, ker_size, sigma, threshhold, harris_k, aperture_size, gridx, gridy);
                Mat res = ImageData::matches(im1.getData(), im2.getData(), kp1, kp2, des1, des2, dst, knn_val);
                imshow("show image", res);
                return true;
            }
            else if (arg_const[1].compare(argv[2]) == 0)
            {
                // int ker_size = (ker_size_tb % 2 == 0) ? ker_size_tb + 1 : ker_size_tb;
                // double thresh = ((double)threshhold_tb) / (MAX_THRESH_BLOB);
                // int gridx = max(1, gridx_tb);
                // int gridy = max(1, gridy_tb);
                im1 = ImageData(argv[4]);
                im2 = ImageData(argv[5]);
                vector<KeyPoint> kp1, kp2;
                Mat des1 = im1.lbpBlob(kp1, ker_size, thresh, gridx, gridy);
                Mat des2 = im2.lbpBlob(kp2, ker_size, thresh, gridx, gridy);
                Mat res = ImageData::matches(im1.getData(), im2.getData(), kp1, kp2, des1, des2, dst, knn_val);
                imshow("show image", res);
                return true;
            }
            else if (arg_const[2].compare(argv[2]) == 0)
            {
                im1 = ImageData(argv[4]);
                im2 = ImageData(argv[5]);
                vector<KeyPoint> kp1, kp2;
                Mat des1 = im1.lbpDoG(kp1);
                Mat des2 = im2.lbpDoG(kp2);
                Mat res = ImageData::matches(im1.getData(), im2.getData(), kp1, kp2, des1, des2, dst, knn_val);
                imshow("show image", res);
                return true;
            }
        }
    }
    return false;
}

void checkAndCreateTrackbar(int argc, char** argv)
{
    if (argc < 2)
        return;
    if (arg_const[3].compare(argv[1]) != 0)
    {
        if (arg_const[0].compare(argv[1]) == 0)
        {
            createTrackbar("kernal size", "show image", &ker_size_tb, MAX_KER_SIZE, TrackbarCallbackFunction);
            createTrackbar("sigma", "show image", &sigma_tb, MAX_SIGMA, TrackbarCallbackFunction);
            createTrackbar("threshhold", "show image", &threshhold_tb, MAX_THRESH, TrackbarCallbackFunction);
            createTrackbar("k value", "show image", &harris_k_param, MAX_K, TrackbarCallbackFunction);
            createTrackbar("aperture size", "show image", &aperture_size_tb, MAX_APERTURE, TrackbarCallbackFunction);
        }
        else if (arg_const[1].compare(argv[1]) == 0)
        {
            createTrackbar("kernal size", "show image", &ker_size_tb, MAX_KER_SIZE, TrackbarCallbackFunction);
            createTrackbar("threshhold", "show image", &threshhold_tb, MAX_THRESH_BLOB, TrackbarCallbackFunction);
        }
        else if (arg_const[2].compare(argv[1]) == 0)
        {
            return;
        }
    }
    else 
    {
        if (arg_const[4].compare(argv[3]) == 0)
        {
            if (arg_const[0].compare(argv[2]) == 0)
            {
                createTrackbar("kernal size", "show image", &ker_size_tb, MAX_KER_SIZE, TrackbarCallbackFunction);
                createTrackbar("sigma", "show image", &sigma_tb, MAX_SIGMA, TrackbarCallbackFunction);
                createTrackbar("threshhold", "show image", &threshhold_tb, MAX_THRESH, TrackbarCallbackFunction);
                createTrackbar("k value", "show image", &harris_k_param, MAX_K, TrackbarCallbackFunction);
                createTrackbar("aperture size", "show image", &aperture_size_tb, MAX_APERTURE, TrackbarCallbackFunction);
                createTrackbar("knn", "show image", &knn, MAX_KNN, TrackbarCallbackFunction);
            }
            else if (arg_const[1].compare(argv[2]) == 0)
            {
                createTrackbar("kernal size", "show image", &ker_size_tb, MAX_KER_SIZE, TrackbarCallbackFunction);
                createTrackbar("threshhold", "show image", &threshhold_tb, MAX_THRESH_BLOB, TrackbarCallbackFunction);
                createTrackbar("knn", "show image", &knn, MAX_KNN, TrackbarCallbackFunction);
            }
            else if (arg_const[2].compare(argv[2]) == 0)
            {
                createTrackbar("knn", "show image", &knn, MAX_KNN, TrackbarCallbackFunction);
                return;
            }
        }
        else if (arg_const[5].compare(argv[3]) == 0)
        {
            if (arg_const[0].compare(argv[2]) == 0)
            {
                createTrackbar("kernal size", "show image", &ker_size_tb, MAX_KER_SIZE, TrackbarCallbackFunction);
                createTrackbar("sigma", "show image", &sigma_tb, MAX_SIGMA, TrackbarCallbackFunction);
                createTrackbar("threshhold", "show image", &threshhold_tb, MAX_THRESH, TrackbarCallbackFunction);
                createTrackbar("k value", "show image", &harris_k_param, MAX_K, TrackbarCallbackFunction);
                createTrackbar("aperture size", "show image", &aperture_size_tb, MAX_APERTURE, TrackbarCallbackFunction);
                createTrackbar("grid size width", "show image", &gridx_tb, MAX_GRID_SIZE, TrackbarCallbackFunction);
                createTrackbar("grid size height", "show image", &gridy_tb, MAX_GRID_SIZE, TrackbarCallbackFunction);
                createTrackbar("knn", "show image", &knn, MAX_KNN, TrackbarCallbackFunction);

            }
            else if (arg_const[1].compare(argv[2]) == 0)
            {
                createTrackbar("kernal size", "show image", &ker_size_tb, MAX_KER_SIZE, TrackbarCallbackFunction);
                createTrackbar("threshhold", "show image", &threshhold_tb, MAX_THRESH_BLOB, TrackbarCallbackFunction);
                createTrackbar("grid size width", "show image", &gridx_tb, MAX_GRID_SIZE, TrackbarCallbackFunction);
                createTrackbar("grid size height", "show image", &gridy_tb, MAX_GRID_SIZE, TrackbarCallbackFunction);
                createTrackbar("knn", "show image", &knn, MAX_KNN, TrackbarCallbackFunction);
            }
            else if (arg_const[2].compare(argv[2]) == 0)
            {
                createTrackbar("grid size width", "show image", &gridx_tb, MAX_GRID_SIZE, TrackbarCallbackFunction);
                createTrackbar("grid size height", "show image", &gridy_tb, MAX_GRID_SIZE, TrackbarCallbackFunction);
                createTrackbar("knn", "show image", &knn, MAX_KNN, TrackbarCallbackFunction);
            }
        }
    }
}

int main(int argc, char** argv)
{
    namedWindow("show image");
    cout << argc << endl;
    ImageData im;
    checkAndCreateTrackbar(argc, argv);
    if (argc == 2)
    {
        isCam = true;
        VideoCapture camera(0);
        if (!camera.isOpened()) {
            cerr << "ERROR: Could not open camera" << endl;
            return 1;
        }
        namedWindow("webcam");
        Mat frame;
        int frame_count = 0;

        while (1) {
        // auto start = chrono::system_clock::now();

            camera >> frame;
            Mat resized;
            resize(frame, resized, Size(MAX_W, MAX_H), INTER_LINEAR);
            imshow("webcam", resized);
            // imshow("webcam", frame);
            if (frame_count == 0)
            {
                handleImage(argv, resized, true);
            }
            
            // cout << frame.rows << " " << frame.cols << endl;
            // auto end = chrono::system_clock::now();
            // chrono::duration<double> diff = end-start;
            // cout << "Time to process last frame (seconds): " << diff.count() 
            //           << " FPS: " << 1.0 / diff.count() << "\n";
            frame_count = (frame_count + 1) % FPS;
            char k = waitKey(20);
            if (k == 'q')
                break;
        }
    }
    else
    {    
        Mat frame;
        int frame_count = 0;
        while (1) {
            // auto start = chrono::system_clock::now();
            if (frame_count == 0)
                handleImage(argv, frame, false);
            
            // cout << frame.rows << " " << frame.cols << endl;
            // auto end = chrono::system_clock::now();
            // chrono::duration<double> diff = end-start;
            // cout << "Time to process last frame (seconds): " << diff.count() 
            //           << " FPS: " << 1.0 / diff.count() << "\n";
            frame_count = (frame_count + 1) % FPS;
            char k = waitKey(20);
            if (k == 'q')
                break;
        }
    }

    return 0;
}

