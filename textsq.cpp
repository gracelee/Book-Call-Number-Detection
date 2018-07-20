


#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>

using namespace cv;
using namespace std;


#define INPUT "/home/cosc/student/dil15/Desktop/Project/template.png"

void detect_text(string input);
int main(int argc, char** argv)
{
    //Mat large = imread(argv[1]);
    detect_text(INPUT);


    return 0;
}

void detect_text(string input){
    Mat large = imread(input);
    Size size(4128,2322);//the dst image size,e.g.100x100
    resize(large,large,size);//resize image

    Mat rgb;
    // downsample and use it for processing
    pyrDown(large, rgb);
    pyrDown(rgb, rgb);
    Mat small;
    cvtColor(rgb, small, CV_BGR2GRAY);
    // morphological gradient
    Mat grad;
    Mat morphKernel = getStructuringElement(MORPH_ELLIPSE, Size(3, 3));
    morphologyEx(small, grad, MORPH_GRADIENT, morphKernel);
    // binarize
    Mat bw;
    threshold(grad, bw, 0.0, 255.0, THRESH_BINARY | THRESH_OTSU);
    // connect horizontally oriented regions
    Mat connected;
    morphKernel = getStructuringElement(MORPH_RECT, Size(9, 1));
    morphologyEx(bw, connected, MORPH_CLOSE, morphKernel);
    // find contours
    Mat mask = Mat::zeros(bw.size(), CV_8UC1);
    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;
    findContours(connected, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
    // filter contours
    for(int idx = 0; idx >= 0; idx = hierarchy[idx][0]){
        Rect rect = boundingRect(contours[idx]);
        Mat maskROI(mask, rect);
        maskROI = Scalar(0, 0, 0);
        // fill the contour
        drawContours(mask, contours, idx, Scalar(255, 255, 255), CV_FILLED);

        RotatedRect rrect = minAreaRect(contours[idx]);
        double r = (double)countNonZero(maskROI) / (rrect.size.width * rrect.size.height);

        Scalar color;
        int thickness = 1;
        // assume at least 25% of the area is filled if it contains text
        if (r > 0.25 &&
        (rrect.size.height > 8 && rrect.size.width > 8) // constraints on region size
        // these two conditions alone are not very robust. better to use something
        //like the number of significant peaks in a horizontal projection as a third condition
        ){
            thickness = 2;
            color = Scalar(0, 255, 0);

        }
        else
        {
            thickness = 1;
            color = Scalar(0, 0, 255);
        }


        Point2f pts[4];
        rrect.points(pts);

        for (int i = 0; i < 4; i++)
        {
            //line(rgb, Point((int)pts[i].x, (int)pts[i].y), Point((int)pts[(i+1)%4].x, (int)pts[(i+1)%4].y), color, thickness);


        }
        Mat sharependCrop;
        Rect lett = rrect.boundingRect();
        Rect bounds(0,0,rgb.cols,rgb.rows);
        Mat crop = rgb(lett & bounds);  // make sure, we stay inside the image
        Size sizeCrop(4128,2322);//the dst image size,e.g.100x100
        resize(crop,crop,sizeCrop);//resize image
        cv::GaussianBlur(crop, crop, cv::Size(0, 0), 3);
        cv::addWeighted(crop, 1.5, crop, -0.5, 0, crop);
        //cv::GaussianBlur(crop, sharependCrop, cv::Size(0,0), 1 );
        //cv::addWeighted(crop, 1.5, sharependCrop, -0.5, 0, crop);
        imwrite(format("./templateOriginal_%d.tiff", idx), crop);


    }

    imwrite("./wholePic.jpg", rgb);
}
