#include "bird.h"

Mat birdseye(const Mat & src) {
   
    Mat dst(src); 

    Point2f objPts[4], imgPts[4];

    objPts[0].x = 0; objPts[0].y = 0; 
    objPts[1].x = 0; objPts[1].y = 5; 
    objPts[2].x = 5; objPts[2].y = 5;
    objPts[3].x = 5; objPts[3].y = 0; 
    
    imgPts[0] = Point2f(100,100);
    imgPts[1] = Point2f(100,600);
    imgPts[2] = Point2f(1000,600);
    imgPts[3] = Point2f(1000,100);

//DRAW THE POINTS in order: B,G,R,YELLOW
/*
    circle(image,imgPts[0],9,Scalar(0,0,255),3);
    circle(image,imgPts[1],9,Scalar(0,255,0),3);
    circle(image,imgPts[2],9,Scalar(255,0,0),3);
    circle(image,imgPts[3],9,Scalar(255,255,0),3);
*/
    Mat H = getPerspectiveTransform(objPts,imgPts);
    
    //USE HOMOGRAPHY TO REMAP THE VIEW
    
    warpPerspective(src,dst,H,src.size(),CV_INTER_LINEAR+CV_WARP_INVERSE_MAP+CV_WARP_FILL_OUTLIERS); 

    return dst;
}
