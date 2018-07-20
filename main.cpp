
#include <opencv2/core/core.hpp>
#include "opencv2/imgproc.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include "highgui.h"
#include <stdlib.h>
#include <stdio.h>


using namespace cv;
using namespace std;
//
//
int erosion_elem = 0;
int erosion_size = 0;
int dilation_elem = 0;
int dilation_size = 0;
int thresh_elem;
int const max_elem = 2;
int const max_kernel_size = 21;
Mat erosion_dst, dilation_dst;
//Mat dst,src,cdst,cdstP;
Mat cdst,cdstP;

Mat thresholdImage;

const int thresh_slider_max = 100;
int alpha_slider;

/// Global variables

int threshold_value = 0;
int threshold_type = 3;;
int const max_value = 255;
int const max_type = 4;
int const max_BINARY_value = 255;

Mat src, src_gray, dst;
char* window_name = "Threshold Demo";

char* trackbar_type = "Type: \n 0: Binary \n 1: Binary Inverted \n 2: Truncate \n 3: To Zero \n 4: To Zero Inverted";
char* trackbar_value = "Value";


/** Function Headers */
void Erosion( int, void* );
void Dilation( int, void* );



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
  erode( src, erosion_dst, element );
  imshow( "Erosion Demo", erosion_dst );
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
  dilate( src, dilation_dst, element );
  imshow( "Dilation Demo", dilation_dst );
}

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


  threshold( src_gray, thresholdImage, threshold_value, max_BINARY_value,threshold_type );

  imshow( window_name, thresholdImage );
  //return thresholdImage;
}

int main(int argc, char** argv ){

  src = imread(argv[1]);
  Size size(4128/3,2322/3);//the dst image size,e.g.100x100
  resize(src,src,size);//resize image
  Mat threshImage;


    if(! src.data )                              // Check for invalid input
    {
        cout <<  "Could not open or find the image" << std::endl ;
        return -1;
    }

    cvtColor(src, src_gray, CV_BGR2GRAY );

    /// Create a window to display results
  namedWindow( window_name, CV_WINDOW_AUTOSIZE );

  /// Create Trackbar to choose type of Threshold
  createTrackbar( trackbar_type,
                  window_name, &threshold_type,
                  max_type, Threshold_Demo );

  createTrackbar( trackbar_value,
                  window_name, &threshold_value,
                  max_value, Threshold_Demo );

  /// Call the function to initialize
  Threshold_Demo( 0, 0 );


    namedWindow( "Erosison Demo", WINDOW_AUTOSIZE );
    namedWindow( "Dilation Demo", WINDOW_AUTOSIZE );
    cvMoveWindow( "Dilation Demo", src.cols, 0 );

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


    //namedWindow( "Display window", WINDOW_AUTOSIZE );// Create a window for display.

    Canny(thresholdImage, dst, 20, 200, 3);
    cvtColor(dst, cdst, CV_GRAY2BGR);
    cdstP = cdst.clone();

    vector<Vec4i> linesP;
  HoughLinesP(dst, linesP, 1, CV_PI/180, 50, 50, 10 );
  for( size_t i = 0; i < linesP.size(); i++ )
  {
    Vec4i l = linesP[i];
    line( cdstP, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0,0,255), 3, CV_AA);
  }


//  Erosion( 0, 0 );
  //  Dilation( 0, 0 );
    //imshow("source", src);

    imshow("Detected Lines (in red) - Probabilistic Line Transform", cdstP);               // Show our image inside it.

    for(;;)
      {
        char c = (char)waitKey( 20 );
        if( c == 27 )
      { break; }
      }                                         // Wait for a keystroke in the window
    return 0;


}
