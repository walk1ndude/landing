
/* This is a standalone program. Pass an image name as a first parameter
 * of the program.  Switch between standard and probabilistic Hough transform
 * by changing "#if 1" to "#if 0" and back */
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>

#define BRIGHT (-1)
#define DARK 1

#define STEGER 0
#define CANNY 1

#include "bird.h"
#include "HoughTransform.h"
#include "StegerLines.h"

using namespace cv;
using namespace std;

int upper_thresh = 100;
int center_thresh = 60;
int line_intersect = 40;
int min_on_line = 15;
int sigma = 40;

int line_length_min2 = 4000;
int line_length_max2 = 30000;

int type = BRIGHT;
int typeC = STEGER;

double angle_perp_delta = M_PI / 16 ;

Mat src,undsrc,edges,color_dst;

string win_name= "Hough Transform";

double angle_pair(Vec6i line_a, Vec6i line_b) {
    double angle_a = atan2(line_a[5],line_a[4]),
           angle_b = atan2(line_b[5],line_b[4]);

    return (angle_a * angle_b <= 0 ) ? fabs(angle_a) + fabs(angle_b) : (angle_a > 0) ? fabs(angle_a - angle_b) : fabs(angle_a + angle_b);
    
}

void find_pattern(vector<Vec6i>&lines, vector<Vec2i>&pair_perp, double angle_delta) {
    for (int line = 0; line != lines.size() - 2 ; line ++) {
        for (int line_pair = line + 1; line_pair != lines.size() - 1 ; line_pair ++ ) {
            double angle = angle_pair(lines[line],lines[line_pair]);
            cout << angle << endl;
            if (angle >= M_PI / 2 - angle_perp_delta && angle <= M_PI / 2 + angle_perp_delta ) {
                pair_perp.push_back(Vec2i(line,line_pair));
            }
        }
    }
}

void hough_transform() {

    if (typeC == STEGER) {
        stegerEdges(undsrc, edges, sigma * 0.1, 0.0, 2, type);
    }
    else {
        Canny(undsrc, edges, 200, 60, 3);
    }
   
    cvtColor(edges, color_dst, CV_GRAY2BGR);

    vector<Vec6i> lines;
        
    HoughLinesGrad(edges, lines, 1, CV_PI/180, max(1,line_intersect), max(1,min_on_line), 10, line_length_min2, line_length_max2);

    vector<Vec2i>pair_perp;

    cout << lines.size() << endl;
 //   find_pattern(lines,pair_perp,angle_perp_delta);

    for (Vec6i Line : lines) {
        line(color_dst, Point(Line[0], Line[1]), Point(Line[2], Line[3]), Scalar(0,0,255), 3, CV_AA);
        cout << "A = " << Line[4] << ", B =  " << Line[5]  << endl;
    }

/*    for (Vec2i Pair : pair_perp) {
        int pair0 = Pair[0], pair1 = Pair[1];
        line(color_dst, Point(lines[pair0][0],lines[pair0][1]), Point(lines[pair0][2],lines[pair0][3]), Scalar(255,0,0), 3, CV_AA);
        line(color_dst, Point(lines[pair1][0],lines[pair1][1]), Point(lines[pair1][2],lines[pair1][3]), Scalar(255,0,0), 3, CV_AA);
        
    }
*/
}
 /*        
        vector<Vec3f> circles;

        HoughCircles(dst, circles, CV_HOUGH_GRADIENT, 1, dst.rows/8, max(1,upper_thresh), max(1,center_thresh), 0, 0);
        
        for (Vec3f Circle : circles) {
            
            Point center(cvRound(Circle[0]),cvRound(Circle[1]));
            int radius = cvRound(Circle[2]);
            
            circle(color_dst, center, 3, Scalar(0,255,0), -1, 8, 0);    
            circle(color_dst, center, radius, Scalar(255,0,0), 3, 8, 0 );
        }*/

void trackbar_callback(int, void*) { 
   hough_transform();
   imshow(win_name, color_dst);
}

int main(int argc, char** argv)
{
    src = imread(argv[1],0);

    type = (!strcmp("-b",argv[2])) ? BRIGHT : (!strcmp("-d",argv[2])) ? DARK : BRIGHT;

    typeC = (!strcmp("-c",argv[3])) ? CANNY : (!strcmp("-s",argv[3])) ? STEGER : CANNY;

    if(argc == 4 && !src.empty()) {

        Mat intrinsics,distortion;

        FileStorage fs("./cameraParams.yml",FileStorage::READ);
        fs["cameraMatrix"] >> intrinsics;
        fs["distCoeffs"] >> distortion;
        fs.release();
        
        undistort(src,undsrc,intrinsics,distortion);

        edges.create(undsrc.size(),CV_8UC1);
        color_dst.create(undsrc.size(),CV_8UC3);

        //color_dst = birdseye(src);
        //dst = birdseye(src);
        
        string out = argv[2];

        if (type == BRIGHT) {
            out = "bright-";
        } else {
            out = "dark-";
        }

        hough_transform();

        namedWindow(win_name, CV_WINDOW_NORMAL);
        setWindowProperty(win_name, CV_WND_PROP_FULLSCREEN, CV_WINDOW_FULLSCREEN);
        imshow(win_name, color_dst);

        createTrackbar("Upper Thresh", win_name, &upper_thresh, 200, trackbar_callback);
        createTrackbar("Center Thresh", win_name, &center_thresh, 200, trackbar_callback);
        createTrackbar("Line Intersect", win_name, &line_intersect, 200, trackbar_callback);
        createTrackbar("Minimum on line", win_name, &min_on_line, 200, trackbar_callback);
        createTrackbar("Gaussian sigma", win_name, &sigma, 100, trackbar_callback);
        createTrackbar("Min length", win_name, &line_length_min2, 50000, trackbar_callback);
        createTrackbar("Max lenght", win_name, &line_length_max2, 50000, trackbar_callback);
 
        waitKey();

        imwrite(out.append(argv[1]), color_dst); 
        return 0;
    }
}
