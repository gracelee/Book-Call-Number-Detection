/**
 * @file Threshold.cpp
 * @brief Sample code that shows how to use the diverse threshold options offered by OpenCV
 * @author OpenCV team
 */

#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/opencv.hpp"
#include <tesseract/baseapi.h>

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <vector>
#include <iostream>
#include <math.h>
#include <string.h>

using namespace cv;
using namespace std;

/// Global variables
int edgeThresh = 1;
int lowThreshold;
int const max_lowThreshold = 100;
int ratio = 3;
int kernel_size = 3;
char* canny_name = "Edge Map";


int erosion_elem = 0;
int erosion_size = 0;
int dilation_elem = 0;
int dilation_size = 0;
int thresh_elem;
int const max_elem = 2;
int const max_kernel_size = 21;

Mat erosion_dst, dilation_dst;
Mat resize_src,outputResult,detected_edges;
Mat HoughLineResult;

Mat cannyResult;

int threshold_value = 0;
int threshold_type = 3;;
int const max_value = 255;
int const max_type = 4;
int const max_BINARY_value = 255;

Mat src, src_gray, dst,src_canny_gray;
char* window_name = "Threshold Demo";

char* trackbar_type = "Type: \n 0: Binary \n 1: Binary Inverted \n 2: Truncate \n 3: To Zero \n 4: To Zero Inverted";
char* trackbar_value = "Value";

/// Function headers
void Threshold_Demo( int, void* );
// the sequence is stored in the specified memory storage
static void findSquares( const Mat&, vector<vector<Point> >&);
static void drawSquares( Mat&, const vector<vector<Point> >& );

/** Function Headers */
void Erosion( int, void* );
void Dilation( int, void* );


/**
 * @function CannyThreshold
 * @brief Trackbar callback - Canny thresholds input with a ratio 1:3
 */
void CannyThreshold(int, void*)
{
  /// Reduce noise with a kernel 3x3
  blur( dst, detected_edges, Size(3,3) );

  /// Canny detector
  Canny( detected_edges, detected_edges, lowThreshold, lowThreshold*ratio, kernel_size );

  /// Using Canny's output as a mask, we display our result
  cannyResult = Scalar::all(0);

  src.copyTo( cannyResult, detected_edges);
  imshow( canny_name, cannyResult );
  imwrite("./result/templatetC.tiff",cannyResult);
 }


/**
 * @function main
 */
int main( int argc, char** argv )
{
  //! [load]
  //Mat grayToColour;
  vector<vector<Point> > squares;

  double thresh = 50;
	double maxValue = 255;


  src = imread( argv[1] ); // Load an image

  Size size(4128/3,2322/3);//the dst image size,e.g.100x100
  resize(src,src,size);//resize image

  if( src.empty() )
    { return -1; }

    //GaussianBlur(inputImage, img_a_blur, Size(5, 5), 0, 0);


  cvtColor( src, src_gray, COLOR_BGR2GRAY ); // Convert the image to Gray
  /// Create a window to display results
  namedWindow( window_name, CV_WINDOW_AUTOSIZE );

  /// Create Trackbar to choose type of Threshold
  createTrackbar( trackbar_type,
                  window_name, &threshold_type,
                  max_type, Threshold_Demo );

  createTrackbar( trackbar_value,
                  window_name, &threshold_value,
                  max_value, Threshold_Demo );
  Threshold_Demo( 0, 0 ); // Call the function to initialize





  /// Wait until user finishes program
  for(;;)
    {
      char c = (char)waitKey( 20 );
      if( c == 27 )
    { break; }
    }

}

int thresh = 50, N = 5;
const char* wndname = "Square Detection Demo";





/**  @function Erosion  */
void Erosion( int, void* )
{
  int erosion_type;
  if( erosion_elem == 0 ){ erosion_type = MORPH_RECT; }
  else if( erosion_elem == 1 ){ erosion_type = MORPH_CROSS; }
  else if( erosion_elem == 2) { erosion_type = MORPH_ELLIPSE; }

  Mat element = getStructuringElement( erosion_type,
                                       Size( 2*erosion_size + 1, 2*erosion_size+1 ),
                                       Point( erosion_size, erosion_size ) );

  /// Apply the erosion operation
  erode( dst, erosion_dst, element );
  imshow( "Erosion Demo", erosion_dst );
  imwrite("./result/originalRectPT/templateE23.tiff",erosion_dst);

  /// Create a matrix of the same type and size as src (for dst)
  cannyResult.create( dst.size(), dst.type() );

  /// Convert the image to grayscale
  //cvtColor( cannyResult, src_canny_gray, CV_BGR2GRAY );

  /// Create a window
  namedWindow( canny_name, CV_WINDOW_AUTOSIZE );

  /// Create a Trackbar for user to enter threshold
  createTrackbar( "Min Threshold:", canny_name, &lowThreshold, max_lowThreshold, CannyThreshold );

  /// Show the image
  CannyThreshold(0, 0);

}

/** @function Dilation */
void Dilation( int, void* )
{
  int dilation_type;
  if( dilation_elem == 0 ){ dilation_type = MORPH_RECT; }
  else if( dilation_elem == 1 ){ dilation_type = MORPH_CROSS; }
  else if( dilation_elem == 2) { dilation_type = MORPH_ELLIPSE; }

  Mat element = getStructuringElement( dilation_type,
                                       Size( 2*dilation_size + 1, 2*dilation_size+1 ),
                                       Point( dilation_size, dilation_size ) );
  /// Apply the dilation operation
  dilate( dst, dilation_dst, element );
  imshow( "Dilation Demo", dilation_dst );
  imwrite("./result/originalRectPT/templateD23.tiff",dilation_dst);
  cannyResult.create( dst.size(), dst.type() );

  /// Convert the image to grayscale
  //cvtColor( dst, src_canny_gray, CV_BGR2GRAY );

  /// Create a window
  namedWindow( canny_name, CV_WINDOW_AUTOSIZE );

  /// Create a Trackbar for user to enter threshold
  createTrackbar( "Min Threshold:", canny_name, &lowThreshold, max_lowThreshold, CannyThreshold );

  /// Show the image
  CannyThreshold(0, 0);
}



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

//s    Mat pyr, timg, gray0(image.size(), CV_8U), gray;

    // down-scale and upscale the image to filter out the noise
    //pyrDown(image, pyr, Size(image.cols/2, image.rows/2));
    //pyrUp(pyr, timg, image.size());


    // blur will enhance edge detection
    Mat timg(image);
    medianBlur(image, timg, 3);
    Mat gray0(timg.size(), CV_8U), gray;

    vector<vector<Point> > contours;

    // find squares in every color plane of the image
    for( int c = 0; c < 3; c++ )
    {
        int ch[] = {c, 0};
        mixChannels(&timg, 1, &gray0, 1, ch, 1);

        // try several threshold levels
        for( int l = 0; l < N; l++ )
        {
            // hack: use Canny instead of zero threshold level.
            // Canny helps to catch squares with gradient shading
            if( l == 0 )
            {
                // apply Canny. Take the upper threshold from slider
                // and set the lower to 0 (which forces edges merging)
                Canny(gray0, gray, 5, thresh, 5);
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
            findContours(gray0, contours, RETR_LIST, CHAIN_APPROX_SIMPLE);

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
                    if( maxCosine < 0.3 )
                        squares.push_back(approx);
                }
            }
        }
    }
}


// the function draws all the squares in the image
static void drawSquares( Mat& image, const vector<vector<Point> >& squares )
{
    for( size_t i = 0; i < squares.size(); i++ )
    {
        const Point* p = &squares[i][0];

        int n = (int)squares[i].size();
        //dont detect the border
        if (p-> x > 3 && p->y > 3)
          polylines(image, &p, &n, 1, true, Scalar(0,255,0), 3, LINE_AA);
    }

    imshow(wndname, image);
}


//![Threshold_Demo]
/**
 * @function Threshold_Demo
 */
 void Threshold_Demo( int, void* )
 {
   /* 0: Binary
      1: Binary Inverted
      2: Threshold Truncated
      3: Threshold to Zero
      4: Threshold to Zero Inverted
    */

   threshold( src_gray, dst, threshold_value, max_BINARY_value,threshold_type );
   imshow( window_name, dst );
   namedWindow( "Erosion Demo", CV_WINDOW_AUTOSIZE );
   namedWindow( "Dilation Demo", CV_WINDOW_AUTOSIZE );
   cvMoveWindow( "Dilation Demo", dst.cols, 0 );

   /// Create Erosion Trackbar
   createTrackbar( "Element:\n 0: Rect \n 1: Cross \n 2: Ellipse", "Erosion Demo",
                   &erosion_elem, max_elem,
                   Erosion );

   createTrackbar( "Kernel size:\n 2n +1", "Erosion Demo",
                   &erosion_size, max_kernel_size,
                   Erosion );

   /// Create Dilation Trackbar
   createTrackbar( "Element:\n 0: Rect \n 1: Cross \n 2: Ellipse", "Dilation Demo",
                   &dilation_elem, max_elem,
                   Dilation );

   createTrackbar( "Kernel size:\n 2n +1", "Dilation Demo",
                   &dilation_size, max_kernel_size,
                   Dilation );
                   imwrite("./result/textsq.tiff",dst);

   /// Default start
   Erosion( 0, 0 );
   Dilation( 0, 0 );



 }
//![Threshold_Demo]
