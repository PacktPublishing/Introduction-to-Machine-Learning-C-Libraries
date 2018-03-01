#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>
#include <stdio.h>

/*
g++ -o fd face_detection.cpp -lopencv_core -lopencv_objdetect -lopencv_highgui -lopencv_videoio -lopencv_imgproc

*/

using namespace std;
using namespace cv;

void detectAndDisplay( Mat & rgb_img, CascadeClassifier & face_classifier)
{
    std::vector<Rect> faces;
    Mat grey_img;
    cvtColor( rgb_img, grey_img, COLOR_BGR2GRAY );
    equalizeHist( grey_img, grey_img );
    
    //here we do the detection 
    face_classifier.detectMultiScale( grey_img, faces, 1.1, 2, 0|CASCADE_SCALE_IMAGE, Size(30, 30) );
    for ( size_t i = 0; i < faces.size(); i++ )
    {
        rectangle(rgb_img, faces[i], Scalar(0,255,0));
    } 

    imshow( "Image", rgb_img );
}

int main( int argc, const char** argv )
{
    CascadeClassifier face_classifier;

    // Load the classifier
    face_classifier.load(argv[1]);

    // load the rgb image
    Mat rgb_img = imread(argv[2]);

    detectAndDisplay(rgb_img, face_classifier);

    waitKey();

    return 0;
}

