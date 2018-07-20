/*
 *  Compile
 *  # g++ txtbin.cpp -o txtbin `pkg-config opencv --cflags --libs`
 *
 *  Get opencv version
 *  # pkg-config --modversion opencv
 *
 *  Run
 *  # ./txtbin input.jpg output.png
 */

#include "string"
#include "fstream"
//#include "/usr/include/opencv2/opencv.hpp"
#include "/usr/include/boost/tuple/tuple.hpp"
#include <opencv2/opencv.hpp>
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

using namespace std;
using namespace cv;
using namespace boost;

void CalcBlockMeanVariance(Mat& Img, Mat& Res, float blockSide=21, float contrast=0.01){
    /*
     *  blockSide: set greater for larger fonts in image and vice versa
     *  contrast: set smaller for lower contrast image
     */

    Mat I;
    Img.convertTo(I, CV_32FC1);
    Res = Mat::zeros(Img.rows / blockSide, Img.cols / blockSide, CV_32FC1);
    Mat inpaintmask;
    Mat patch;
    Mat smallImg;
    Scalar m, s;

    for(int i = 0; i < Img.rows - blockSide; i += blockSide){
        for(int j = 0; j < Img.cols - blockSide; j += blockSide){
            patch = I(Range(i, i + blockSide + 1), Range(j, j + blockSide + 1));
            meanStdDev(patch, m, s);

            if(s[0] > contrast){
                Res.at<float>(i / blockSide, j / blockSide) = m[0];
            }
            else{
                Res.at<float>(i / blockSide, j / blockSide) = 0;
            }
        }
    }

    resize(I, smallImg, Res.size());

    threshold(Res, inpaintmask, 0.02, 1.0, THRESH_BINARY);

    Mat inpainted;
    smallImg.convertTo(smallImg, CV_8UC1, 255);

    inpaintmask.convertTo(inpaintmask, CV_8UC1);
    inpaint(smallImg, inpaintmask, inpainted, 5, INPAINT_TELEA);

    resize(inpainted, Res, Img.size());
    Res.convertTo(Res, CV_32FC1, 1.0 / 255.0);
}

tuple<int, int, int, int> detect_text_box(string input, Mat& res, bool draw_contours=false){
    Mat large = imread(input);

    bool test_output = false;

    int
        top = large.rows,
        bottom = 0,
        left = large.cols,
        right = 0;

    int
        rect_bottom,
        rect_right;

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

    Scalar color = Scalar(0, 255, 0);
    Scalar color2 = Scalar(0, 0, 255);
    int thickness = 2;

    // filter contours
    for(int idx = 0; idx >= 0; idx = hierarchy[idx][0]){
        Rect rect = boundingRect(contours[idx]);
        Mat maskROI(mask, rect);
        maskROI = Scalar(0, 0, 0);
        // fill the contour
        drawContours(mask, contours, idx, Scalar(255, 255, 255), CV_FILLED);
        // ratio of non-zero pixels in the filled region
        double r = (double)countNonZero(maskROI) / (rect.width * rect.height);

        // assume at least 25% of the area is filled if it contains text
        if (r > 0.25 &&
        (rect.height > 8 && rect.width > 8) // constraints on region size
        // these two conditions alone are not very robust. better to use something
        //like the number of significant peaks in a horizontal projection as a third condition
        ){
            if(draw_contours){
                rectangle(res, Rect(rect.x * 4, rect.y * 4, rect.width * 4, rect.height * 4), color, thickness);
            }

            if(test_output){
                rectangle(rgb, rect, color, thickness);
            }

            if(rect.y < top){
                top = rect.y;
            }
            rect_bottom = rect.y + rect.height;
            if(rect_bottom > bottom){
                bottom = rect_bottom;
            }
            if(rect.x < left){
                left = rect.x;
            }
            rect_right = rect.x + rect.width;
            if(rect_right > right){
                right = rect_right;
            }
        }
    }

    if(draw_contours){
        rectangle(res, Point(left * 4, top * 4), Point(right * 4, bottom * 4), color2, thickness);
    }

    if(test_output){
        rectangle(rgb, Point(left, top), Point(right, bottom), color2, thickness);
        imwrite(string("test_text_contours.jpg"), rgb);
    }

    return make_tuple(left * 4, top * 4, (right - left) * 4, (bottom - top) * 4);
}

int main(int argc, char* argv[]){
    string input;
    string output = "output.png";

    int
        width = 0,
        height = 0,
        blockside = 9;

    bool
        crop = false,
        draw = false;

    float margin = 0;

    cout << "OpenCV version: " << CV_VERSION << endl;

    //  Return error if arguments are missing
    if(argc < 3){
        cerr << "\nUsage: txtbin input [options] output\n\n"
            "Options:\n"
            "\t-w <number>          -- set max width (keeps aspect ratio)\n"
            "\t-h <number>          -- set max height (keeps aspect ratio)\n"
            "\t-c                   -- crop text content contour\n"
            "\t-m <number>          -- add margins (number in %)\n"
            "\t-b <number>          -- set blockside\n"
            "\t-d                   -- draw text content contours (debugging)\n" << endl;
        return 1;
    }

    //  Parse arguments
    for(int i = 1; i < argc; i++){
        if(i == 1){
            input = string(argv[i]);

            //  Return error if input file is invalid
            ifstream stream(input.c_str());
            if(!stream.good()){
                cerr << "Error: Input file is invalid!" << endl;
                return 1;
            }
        }
        else if(string(argv[i]) == "-w"){
            width = atoi(argv[++i]);
        }
        else if(string(argv[i]) == "-h"){
            height = atoi(argv[++i]);
        }
        else if(string(argv[i]) == "-c"){
            crop = true;
        }
        else if(string(argv[i]) == "-m"){
            margin = atoi(argv[++i]);
        }
        else if(string(argv[i]) == "-b"){
            blockside = atoi(argv[++i]);
        }
        else if(string(argv[i]) == "-d"){
            draw = true;
        }
        else if(i == argc - 1){
            output = string(argv[i]);
        }
    }

    Mat Img = imread(input, CV_LOAD_IMAGE_GRAYSCALE);
    Mat res;
    Img.convertTo(Img, CV_32FC1, 1.0 / 255.0);
    CalcBlockMeanVariance(Img, res, blockside);
    res = 1.0 - res;
    res = Img + res;
    threshold(res, res, 0.85, 1, THRESH_BINARY);

    int
        txt_x,
        txt_y,
        txt_width,
        txt_height;

    if(crop || draw){
        tie(txt_x, txt_y, txt_width, txt_height) = detect_text_box(input, res, draw);
    }

    if(crop){
        //res = res(Rect(txt_x, txt_y, txt_width, txt_height)).clone();
        res = res(Rect(txt_x, txt_y, txt_width, txt_height));
    }

    if(margin){
        int border = res.cols * margin / 100;
        copyMakeBorder(res, res, border, border, border, border, BORDER_CONSTANT, Scalar(255, 255, 255));
    }

    float
        width_input = res.cols,
        height_input = res.rows;

    bool resized = false;

    //  Downscale image
    if(width > 0 && width_input > width){
        float scale = width_input / width;
        width_input /= scale;
        height_input /= scale;
        resized = true;
    }
    if(height > 0 && height_input > height){
        float scale = height_input / height;
        width_input /= scale;
        height_input /= scale;
        resized = true;
    }
    if(resized){
        resize(res, res, Size(round(width_input), round(height_input)));
    }

    imwrite(output, res * 255);

    return 0;
}
