#include "ImageData.h"

using namespace cv;
using namespace std;
const string arg_const[] = {"harris", "blob", "dog", "m", "sift", "lbp"};


void TrackbarCallbackFunction(int pos, void* userData)
{
    int valueFromUser = *(static_cast<int*>(userData));
}

int main(int argc, char** argv)
{
    namedWindow("show image");
    // namedWindow("show image input");
    ImageData im;

    // imshow("show image input", im.getData());
    if (arg_const[3].compare(argv[0]) != 0)
    {
        if (arg_const[0].compare(argv[1]) == 0)
        {
            im = ImageData(argv[2]);
            Mat res = im.harrisDectect(3, 1.4, 0.04, 165, 1);
            imshow("show image", res);
            waitKey(0);
        }
        else if (arg_const[1].compare(argv[1]) == 0)
        {
            im = ImageData(argv[2]);
            Mat res = im.blobDetect(19, 0.4);
            imshow("show image", res);
            waitKey(0);
        }
        else if (arg_const[2].compare(argv[1]) == 0)
        {
            im = ImageData(argv[2]);
            Mat res = im.DoGDetect();
            imshow("show image", res);
            waitKey(0);
        }
    }
    else 
    {
        ImageData im1, im2;
        Mat dst;
        if (arg_const[4].compare(argv[3]) == 0)
        {
            if (arg_const[0].compare(argv[1]) == 0)
            {
                im1 = ImageData(argv[4]);
                im2 = ImageData(argv[5]);
                vector<KeyPoint> kp1, kp2;
                Mat des1 = im1.Sift(kp1);
                Mat des2 = im2.Sift(kp2);
                Mat res = ImageData::matches(im1.getData(), im2.getData(), kp1, kp2, des1, des2, dst, 2);
                imshow("show image", res);
                waitKey(0);
            }
            else if (arg_const[1].compare(argv[1]) == 0)
            {
                im1 = ImageData(argv[4]);
                im2 = ImageData(argv[5]);
                vector<KeyPoint> kp1, kp2;
                Mat des1 = im1.SiftBlob(kp1, 19, 0.4);
                Mat des2 = im2.SiftBlob(kp2, 19, 0.4);
                Mat res = ImageData::matches(im1.getData(), im2.getData(), kp1, kp2, des1, des2, dst, 2);
                imshow("show image", res);
                waitKey(0);
            }
            else if (arg_const[2].compare(argv[1]) == 0)
            {
                im1 = ImageData(argv[4]);
                im2 = ImageData(argv[5]);
                vector<KeyPoint> kp1, kp2;
                Mat des1 = im1.DoGSift(kp1);
                Mat des2 = im2.DoGSift(kp2);
                Mat res = ImageData::matches(im1.getData(), im2.getData(), kp1, kp2, des1, des2, dst, 2);
                imshow("show image", res);
                waitKey(0);
            }
        }
        else if (arg_const[4].compare(argv[4]))
        {
            if (arg_const[0].compare(argv[1]) == 0)
            {
                im1 = ImageData(argv[4]);
                im2 = ImageData(argv[5]);
                vector<KeyPoint> kp1, kp2;
                Mat des1 = im1.lbp(kp1);
                Mat des2 = im2.lbp(kp2);
                Mat res = ImageData::matches(im1.getData(), im2.getData(), kp1, kp2, des1, des2, dst, 2);
                imshow("show image", res);
                waitKey(0);
            }
            else if (arg_const[1].compare(argv[1]) == 0)
            {
                im1 = ImageData(argv[4]);
                im2 = ImageData(argv[5]);
                vector<KeyPoint> kp1, kp2;
                Mat des1 = im1.lbpBlob(kp1, 19, 0.4);
                Mat des2 = im2.lbpBlob(kp2, 19, 0.4);
                Mat res = ImageData::matches(im1.getData(), im2.getData(), kp1, kp2, des1, des2, dst, 2);
                imshow("show image", res);
                waitKey(0);
            }
            else if (arg_const[2].compare(argv[1]) == 0)
            {
                im1 = ImageData(argv[4]);
                im2 = ImageData(argv[5]);
                vector<KeyPoint> kp1, kp2;
                Mat des1 = im1.lbpDoG(kp1);
                Mat des2 = im2.lbpDoG(kp2);
                Mat res = ImageData::matches(im1.getData(), im2.getData(), kp1, kp2, des1, des2, dst, 2);
                imshow("show image", res);
                waitKey(0);
            }
        }
    }

    return 0;
}

