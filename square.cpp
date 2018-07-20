// The "Square Detector" program.
// It loads several images sequentially and tries to find squares in
// each image
//#define VIDEO_CAPTURE


#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <tesseract/baseapi.h>
#include <leptonica/allheaders.h>
#include <sys/time.h>

#include <iostream>
#include <math.h>
#include<sstream>
#include <string.h>

using namespace cv;
using namespace std;

static void help()
{
    cout <<
    "\nA program using pyramid scaling, Canny, contours, contour simpification and\n"
    "memory storage to find squares in a list of images\n"
    "Returns sequence of squares detected on the image.\n"
    "the sequence is stored in the specified memory storage\n"
    "Call:\n"
    "./squares\n"
    "Using OpenCV version %s\n" << CV_VERSION << "\n" << endl;
}


int thresh = 50, N = 5;
const char* wndname = "Square Detection Demo";

// helper function:
// finds a cosine of angle between vectors
// from pt0->pt1 and from pt0->pt2
static double angle( Point pt1, Point pt2, Point pt0 )
{
    double dx1 = pt1.x - pt0.x;
    double dy1 = pt1.y - pt0.y;
    double dx2 = pt2.x - pt0.x;
    double dy2 = pt2.y - pt0.y;
    return (dx1*dx2 + dy1*dy2)/sqrt((dx1*dx1 + dy1*dy1)*(dx2*dx2 + dy2*dy2) + 1e-10);
}

// returns sequence of squares detected on the image.
// the sequence is stored in the specified memory storage
static void findSquares( const Mat& image, vector<vector<Point> >& squares )
{
    squares.clear();
    // blur will enhance edge detection
    Mat timg(image);
    medianBlur(image, timg, 9);
    Mat gray0(timg.size(), CV_8U), gray;

    vector<vector<Point> > contours;

    // // find squares in every color plane of the image
    for( int c = 0; c < 3; c++ )
    {
        int ch[] = {c, 0};
        mixChannels(&timg, 1, &gray0, 1, ch, 1);
        ;
        // try several threshold levels
        for( int l = 0; l < N; l++ )
        {
            // hack: use Canny instead of zero threshold level.
            // Canny helps to catch squares with gradient shading
            if( l == 0 )
            {
                // apply Canny. Take the upper threshold from slider
                // and set the lower to 0 (which forces edges merging)
                Canny(gray0, gray, 0, thresh, 5);


                // dilate canny output to remove potential
                // holes between edge segments
                dilate(gray, gray, Mat(), Point(-1,-1));


            }
            else
            {
                // apply threshold if l!=0:
                //     tgray(x,y) = gray(x,y) < (l+1)*255/N ? 255 : 0
                gray = gray0 >= (l+1)*255/N;

            }

            // find contours and store them all as a list
            findContours(gray, contours, RETR_LIST, CHAIN_APPROX_SIMPLE);


            vector<Point> approx;

            // test each contour
            for( size_t i = 0; i < contours.size(); i++ )
            {
                // approximate contour with accuracy proportional
                // to the contour perimeter
                approxPolyDP(Mat(contours[i]), approx, arcLength(Mat(contours[i]), true)*0.02, true);


                // square contours should have 4 vertices after approximation
                // relatively large area (to filter out noisy contours)
                // and be convex.
                // Note: absolute value of an area is used because
                // area may be positive or negative - in accordance with the
                // contour orientation
                if( approx.size() == 4 &&
                    fabs(contourArea(Mat(approx))) > 1000 &&
                    isContourConvex(Mat(approx)) )
                {
                    double maxCosine = 0;

                    for( int j = 2; j < 5; j++ )
                    {
                        // find the maximum cosine of the angle between joint edges
                        double cosine = fabs(angle(approx[j%4], approx[j-2], approx[j-1]));
                        maxCosine = MAX(maxCosine, cosine);
                    }

                    // if cosines of all angles are small
                    // (all angles are ~90 degree) then write quandrange
                    // vertices to resultant sequence
                    if( maxCosine < 0.3 ){ // If the angle is more than 72 degree
                        squares.push_back(approx);
                      }
                }
            }
        }
    }
}


// the function draws all the squares in the image
static void drawSquares( Mat& image, const vector<vector<Point> >& squares )
{
   //cout << "square size " << squares.size();

    for( size_t i = 0; i < squares.size(); i++ )
    {
        const Point* p = &squares[i][0];

        int n = (int)squares[i].size();

        //dont detect the border
        if (p-> x > 3 && p->y > 3){
          polylines(image, &p, &n, 1, true, Scalar(255,255,0), 3, LINE_AA);
        }

    }

    imshow(wndname, image);

}

void detect_text(string input){
    Mat large = imread(input);

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
            line(rgb, Point((int)pts[i].x, (int)pts[i].y), Point((int)pts[(i+1)%4].x, (int)pts[(i+1)%4].y), color, thickness);


        }
        Rect lett = rrect.boundingRect();
        Rect bounds(0,0,rgb.cols,rgb.rows);
        Mat crop = rgb(lett & bounds);  // make sure, we stay inside the image
        imwrite(format("letters_%04d.png", idx), crop);

    }

    imwrite("yesy.jpg", rgb);
}


int main(int argc, char** argv)
{
#ifdef VIDEO_CAPTURE
        VideoCapture capture(0);
        Mat frame;

        if( !capture.isOpened() )
            throw "No video detected";

        //namedWindow( "w", 1);
        for( ; ; )
        {
            capture >> frame;
            if(frame.empty())
                break;

                vector<vector<Point> > squares;
                  namedWindow( wndname, 1 );

                  tesseract::TessBaseAPI *myOCR =
                  new tesseract::TessBaseAPI();

                  Mat originalImage;

                  if (myOCR->Init(NULL, "eng")) {
                    fprintf(stderr, "Could not initialize tesseract.\n");
                    exit(1);
                  }

                  char filename[80];
                  imwrite("originalImage.jpg", frame);
                  frame.copyTo(originalImage);
                  findSquares(frame, squares);

                  for (size_t i=0; i<squares.size(); i++) {
                      Rect r = boundingRect(squares[i]);
                      Mat crop = originalImage(r);
                      cv::cvtColor(crop, crop, CV_BGR2GRAY);
                      PIX *pix = pixCreate(crop.size().width, crop.size().height, 8);
                      for(int i=0; i<crop.rows; i++){
                          for(int j=0; j<crop.cols; j++){
                            pixSetPixel(pix, j,i, (l_uint32) crop.at<uchar>(i,j));
                          }
                        }
                      myOCR->SetImage(pix);
                      char* outText = myOCR->GetUTF8Text();
                      printf("OCR output:\n");
                      printf(outText);
                      myOCR->Clear();
                      myOCR->End();
                      delete [] outText;
                      pixDestroy(&pix);
                      if (myOCR->Init(NULL, "eng")) {
                        fprintf(stderr, "Could not initialize tesseract.\n");
                        exit(1);
                      }

                    }

            drawSquares(frame, squares);
            imshow("video", frame);
            waitKey(5); // waits to display frame
        }
        waitKey(0); // key press to close window

#else
      vector<vector<Point> > squares;
      namedWindow( wndname, 1 );

      tesseract::TessBaseAPI *myOCR =
      new tesseract::TessBaseAPI();

      Mat image = imread(argv[1], 1);
      if( image.empty() )
      {
          cout << "Couldn't load " << argv[1] << endl;
          //continue;
      }
      Mat originalImage;

      if (myOCR->Init(NULL, "eng")) {
        fprintf(stderr, "Could not initialize tesseract.\n");
        exit(1);
      }

      char filename[80];
      imwrite("originalImage.jpg", image);
      //Size size(4128/3,2322/3);//the dst image size,e.g.100x100
      image.copyTo(originalImage);
      // resize(image,image,size);//resize image
    //  resize(image,originalImage,size);//resize image
      findSquares(image, squares);

      for (size_t i=0; i<squares.size(); i++) {
          Rect r = boundingRect(squares[i]);
          Mat crop = originalImage(r);
          cv::cvtColor(crop, crop, CV_BGR2GRAY);
          PIX *pix = pixCreate(crop.size().width, crop.size().height, 8);
          for(int i=0; i<crop.rows; i++){
              for(int j=0; j<crop.cols; j++){
                pixSetPixel(pix, j,i, (l_uint32) crop.at<uchar>(i,j));
              }
            }
          myOCR->SetImage(pix);
          char* outText = myOCR->GetUTF8Text();
          printf("OCR output:\n");
          printf(outText);
          myOCR->Clear();
          myOCR->End();
          delete [] outText;
          pixDestroy(&pix);
          if (myOCR->Init(NULL, "eng")) {
            fprintf(stderr, "Could not initialize tesseract.\n");
            exit(1);
          }

        }

      drawSquares(originalImage, squares);
      imwrite("squarefoud_2.jpg", originalImage);

      for(;;)
        {
          char c = (char)waitKey( 20 );
          if( c == 27 )
        { break; }
      }

    return 0;

#endif
}
