/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000, Intel Corporation, all rights reserved.
// Copyright (C) 2013, OpenCV Foundation, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#include "HoughTransform.h"
#include <math.h>

/*
Here image is an input raster;
step is it's step; size characterizes it's ROI;
rho and theta are discretization steps (in pixels and radians correspondingly).
threshold is the minimum number of pixels in the feature for it
to be a candidate for line. lines is the output
array of (rho, theta) pairs. linesMax is the buffer size (number of pairs).
Functions return the actual number of found lines.
*/
/****************************************************************************************\
*                              Probabilistic Hough Transform                             *
\****************************************************************************************/

int line_lenght2(Vec6i line){
    return (line[0] - line[2]) * (line[0] - line[2]) + (line[1] - line[3]) * (line[1] - line[3]); 
}


static void
HoughLinesGradient( Mat& image,
                         float rho, float theta, int threshold,
                         int lineLength, int lineGap,
                         std::vector<Vec6i>& lines, int linesMax,
                         int line_length_min2, int line_length_max2 )
{
    Point pt;
    float irho = 1 / rho;
    RNG rng((uint64)-1);

    CV_Assert( image.type() == CV_8UC1 );

    int width = image.cols;
    int height = image.rows;

    int numangle = cvRound(CV_PI / theta);
    int numrho = cvRound(((width + height) * 2 + 1) / rho);

    Mat accum = Mat::zeros( numangle, numrho, CV_32SC1 );
    Mat mask( height, width, CV_8UC1 );
    std::vector<float> trigtab(numangle*2);

    for( int n = 0; n < numangle; n++ )
    {
        trigtab[n*2] = (float)(cos((double)n*theta) * irho);
        trigtab[n*2+1] = (float)(sin((double)n*theta) * irho);
    }
    const float* ttab = &trigtab[0];
    uchar* mdata0 = mask.data;
    std::vector<Point> nzloc;

    // stage 1. collect non-zero image points
    for( pt.y = 0; pt.y < height; pt.y++ )
    {
        const uchar* data = image.ptr(pt.y);
        uchar* mdata = mask.ptr(pt.y);
        for( pt.x = 0; pt.x < width; pt.x++ )
        {
            if( data[pt.x] > 50)
            {
                mdata[pt.x] = (uchar)1;
                nzloc.push_back(pt);
            }
            else
                mdata[pt.x] = 0;
        }
    }

    int count = (int)nzloc.size();
    const uchar* data = image.data;
 

    // stage 2. process all the points in random order
    for( ; count > 0; count-- )
    {
        // choose random point out of the remaining ones
        int idx = rng.uniform(0, count);
        int max_val = threshold-1, max_n = 0;
        Point point = nzloc[idx];
        Point line_end[2];
        float a, b;
        int* adata = (int*)accum.data;
        int i = point.y, j = point.x, k, x0, y0, dx0, dy0, xflag;
        int good_line;
        const int shift = 16;

        // "remove" it by overriding it with the last element
        nzloc[idx] = nzloc[count-1];

        // check if it has been excluded already (i.e. belongs to some other line)
        if( !mdata0[i*width + j] )
            continue;

        //find image gradient in point (i,j)
        int gradX,gradY;
        
        if ( i == 0 ) {
            
            if ( j ==  0) {
                gradX = -10 * data[(i + 1) * width + j] - 3 * data[(i + 1) * width + (j + 1)];
                gradY = 10 * data[i * width + (j + 1)] + 3 * data[(i + 1) * width + (j + 1)];
            }
            else if ( j == height ) {
                gradX = -3 * data[(i + 1) * width + (j - 1)] - 10 * data[(i + 1) * width + j];
                gradY = -10 * data[i * width + (j - 1)] - 10 * data[(i + 1) * width + (j - 1)];
            }
            else {
                gradX = -3 * data[(i + 1) * width + (j - 1)] - 10 * data[(i + 1) * width + j] - 3 * data[(i + 1) * width + (j + 1)];
                gradY = -10 * data[i * width + (j - 1)] - 3 * data[(i + 1) * width + (j - 1)] + 10 * data[i * width + (j + 1)] + 3 * data[(i + 1) * width + (j + 1)];           
            }

        }
        else if ( i == width ) {

            if ( j ==  0) {
                gradX = 10 * data[(i - 1) * width + j] + 3 * data[(i - 1) * width + (j + 1)];
                gradY = 3 * data[(i - 1) * width + (j + 1)] + 10 * data[i * width + (j + 1)];
            }
            else if ( j == height ) {
                gradX = 3 * data[(i - 1) * width + (j - 1)] + 10 * data[(i - 1) * width + j];
                gradY = -3 * data[(i - 1) * width + (j - 1)] - 10 * data[i  * width + (j - 1)];
            }
            else {
                gradX = 3 * data[(i - 1) * width + (j - 1)] + 10 * data[(i - 1) * width + j] + 3 * data[(i - 1) * width + (j + 1)];
                gradY = -10 * data[i * width + (j - 1)] - 3 * data[(i - 1) * width + (j - 1)] + 10 * data[i * width + (j + 1)] + 3 * data[(i + 1) * width + (j + 1)];           
            }

        }
        else {
       
            gradX = 3 * data[(i - 1) * width + (j - 1)] + 10 * data[(i - 1) * width + j] + 3 * data[(i - 1) * width + (j + 1)] -
                    3 * data[(i + 1) * width + (j - 1)] - 10 * data[(i + 1) * width + j] - 3 * data[(i + 1) * width + (j + 1)];
            
            gradY = 3 * data[(i - 1) * width + (j + 1)] + 10 * data[i * width + (j + 1)] + 3 * data[(i + 1) * width + (j + 1)] -
                    3 * data[(i - 1) * width + (j - 1)] - 10 * data[i * width + (j - 1)] - 3 * data[(i + 1) * width + (j - 1)];

        }

        int gradTheta = cvRound(atan2(gradY,gradX) * 180 / CV_PI);

        if (gradTheta < 0) {
            gradTheta += 180;
        }
 
        // update accumulator, find the most probable line
        for( int n = max(gradTheta - 15,0); n < min(numangle,gradTheta + 15); n++, adata += numrho )
        {
            int r = cvRound( j * ttab[n*2] + i * ttab[n*2+1] );
            r += (numrho - 1) / 2;
            int val = ++adata[r];
            if( max_val < val )
            {
                max_val = val;
                max_n = n;
            }
        }

        // if it is too "weak" candidate, continue with another point
        if( max_val < threshold )
            continue;

        // from the current point walk in each direction
        // along the found line and extract the line segment
        a = -ttab[max_n * 2 + 1];
        b = ttab[max_n * 2];
        x0 = j;
        y0 = i;
        if( fabs(a) > fabs(b) )
        {
            xflag = 1;
            dx0 = a > 0 ? 1 : -1;
            dy0 = cvRound( b*(1 << shift)/fabs(a) );
            y0 = (y0 << shift) + (1 << (shift-1));
        }
        else
        {
            xflag = 0;
            dy0 = b > 0 ? 1 : -1;
            dx0 = cvRound( a*(1 << shift)/fabs(b) );
            x0 = (x0 << shift) + (1 << (shift-1));
        }

        for( k = 0; k < 2; k++ )
        {
            int gap = 0, x = x0, y = y0, dx = dx0, dy = dy0;

            if( k > 0 )
                dx = -dx, dy = -dy;

            // walk along the line using fixed-point arithmetics,
            // stop at the image border or in case of too big gap
            for( ;; x += dx, y += dy )
            {
                uchar* mdata;
                int i1, j1;

                if( xflag )
                {
                    j1 = x;
                    i1 = y >> shift;
                }
                else
                {
                    j1 = x >> shift;
                    i1 = y;
                }

                if( j1 < 0 || j1 >= width || i1 < 0 || i1 >= height )
                    break;

                mdata = mdata0 + i1*width + j1;

                // for each non-zero point:
                //    update line end,
                //    clear the mask element
                //    reset the gap
                if( *mdata )
                {
                    gap = 0;
                    line_end[k].y = i1;
                    line_end[k].x = j1;
                }
                else if( ++gap > lineGap )
                    break;
            }
        }

        good_line = std::abs(line_end[1].x - line_end[0].x) >= lineLength ||
                    std::abs(line_end[1].y - line_end[0].y) >= lineLength;

        for( k = 0; k < 2; k++ )
        {
            int x = x0, y = y0, dx = dx0, dy = dy0;

            if( k > 0 )
                dx = -dx, dy = -dy;

            // walk along the line using fixed-point arithmetics,
            // stop at the image border or in case of too big gap
            for( ;; x += dx, y += dy )
            {
                uchar* mdata;
                int i1, j1;

                if( xflag )
                {
                    j1 = x;
                    i1 = y >> shift;
                }
                else
                {
                    j1 = x >> shift;
                    i1 = y;
                }

                mdata = mdata0 + i1*width + j1;

                // for each non-zero point:
                //    update line end,
                //    clear the mask element
                //    reset the gap
                if( *mdata )
                {
                    if( good_line )
                    {
                        adata = (int*)accum.data;
                        for( int n = 0; n < numangle; n++, adata += numrho )
                        {
                            int r = cvRound( j1 * ttab[n*2] + i1 * ttab[n*2+1] );
                            r += (numrho - 1) / 2;
                            adata[r]--;
                        }
                    }
                    *mdata = 0;
                }

                if( i1 == line_end[k].y && j1 == line_end[k].x )
                    break;
            }
        }

        if( good_line )
        {
            Vec6i lr(line_end[0].x,
                    line_end[0].y,
                    line_end[1].x,
                    line_end[1].y,
                    line_end[1].y - line_end[0].y,
                    line_end[0].x - line_end[1].x);

            int lengthL = line_lenght2(lr);
            if (lengthL >= line_length_min2 && lengthL <= line_length_max2) {
                lines.push_back(lr);
            }

            if( (int)lines.size() >= linesMax )
                return;
        }
    }
}

void HoughLinesGrad(InputArray _image, OutputArray _lines,
                     double rho, double theta, int threshold,
                     double minLineLength, double maxGap,
                     int line_length_min2, int line_length_max2)
{
    Mat image = _image.getMat();
    std::vector<Vec6i> lines;
    HoughLinesGradient(image, (float)rho, (float)theta, threshold, cvRound(minLineLength), cvRound(maxGap), lines, INT_MAX, line_length_min2, line_length_max2);
    Mat(lines).copyTo(_lines);
}

/****************************************************************************************\
*                                     Circle Detection                                   *
\****************************************************************************************/
/*
static void
icvHoughCirclesGradient( CvMat* img, float dp, float min_dist,
                         int min_radius, int max_radius,
                         int canny_threshold, int acc_threshold,
                         CvSeq* circles, int circles_max )
{
    const int SHIFT = 10, ONE = 1 << SHIFT;
    cv::Ptr<CvMat> dx, dy;
    cv::Ptr<CvMat> edges, accum, dist_buf;
    std::vector<int> sort_buf;
    cv::Ptr<CvMemStorage> storage;

    int x, y, i, j, k, center_count, nz_count;
    float min_radius2 = (float)min_radius*min_radius;
    float max_radius2 = (float)max_radius*max_radius;
    int rows, cols, arows, acols;
    int astep, *adata;
    float* ddata;
    CvSeq *nz, *centers;
    float idp, dr;
    CvSeqReader reader;

    edges = cvCreateMat( img->rows, img->cols, CV_8UC1 );
    cvCanny( img, edges, MAX(canny_threshold/2,1), canny_threshold, 3 );

    dx = cvCreateMat( img->rows, img->cols, CV_16SC1 );
    dy = cvCreateMat( img->rows, img->cols, CV_16SC1 );
    cvSobel( img, dx, 1, 0, 3 );
    cvSobel( img, dy, 0, 1, 3 );

    if( dp < 1.f )
        dp = 1.f;
    idp = 1.f/dp;
    accum = cvCreateMat( cvCeil(img->rows*idp)+2, cvCeil(img->cols*idp)+2, CV_32SC1 );
    cvZero(accum);

    storage = cvCreateMemStorage();
    nz = cvCreateSeq( CV_32SC2, sizeof(CvSeq), sizeof(CvPoint), storage );
    centers = cvCreateSeq( CV_32SC1, sizeof(CvSeq), sizeof(int), storage );

    rows = img->rows;
    cols = img->cols;
    arows = accum->rows - 2;
    acols = accum->cols - 2;
    adata = accum->data.i;
    astep = accum->step/sizeof(adata[0]);
    // Accumulate circle evidence for each edge pixel
    for( y = 0; y < rows; y++ )
    {
        const uchar* edges_row = edges->data.ptr + y*edges->step;
        const short* dx_row = (const short*)(dx->data.ptr + y*dx->step);
        const short* dy_row = (const short*)(dy->data.ptr + y*dy->step);

        for( x = 0; x < cols; x++ )
        {
            float vx, vy;
            int sx, sy, x0, y0, x1, y1, r;
            CvPoint pt;

            vx = dx_row[x];
            vy = dy_row[x];

            if( !edges_row[x] || (vx == 0 && vy == 0) )
                continue;

            float mag = std::sqrt(vx*vx+vy*vy);
            assert( mag >= 1 );
            sx = cvRound((vx*idp)*ONE/mag);
            sy = cvRound((vy*idp)*ONE/mag);

            x0 = cvRound((x*idp)*ONE);
            y0 = cvRound((y*idp)*ONE);
            // Step from min_radius to max_radius in both directions of the gradient
            for(int k1 = 0; k1 < 2; k1++ )
            {
                x1 = x0 + min_radius * sx;
                y1 = y0 + min_radius * sy;

                for( r = min_radius; r <= max_radius; x1 += sx, y1 += sy, r++ )
                {
                    int x2 = x1 >> SHIFT, y2 = y1 >> SHIFT;
                    if( (unsigned)x2 >= (unsigned)acols ||
                        (unsigned)y2 >= (unsigned)arows )
                        break;
                    adata[y2*astep + x2]++;
                }

                sx = -sx; sy = -sy;
            }

            pt.x = x; pt.y = y;
            cvSeqPush( nz, &pt );
        }
    }

    nz_count = nz->total;
    if( !nz_count )
        return;
    //Find possible circle centers
    for( y = 1; y < arows - 1; y++ )
    {
        for( x = 1; x < acols - 1; x++ )
        {
            int base = y*(acols+2) + x;
            if( adata[base] > acc_threshold &&
                adata[base] > adata[base-1] && adata[base] > adata[base+1] &&
                adata[base] > adata[base-acols-2] && adata[base] > adata[base+acols+2] )
                cvSeqPush(centers, &base);
        }
    }

    center_count = centers->total;
    if( !center_count )
        return;

    sort_buf.resize( MAX(center_count,nz_count) );
    cvCvtSeqToArray( centers, &sort_buf[0] );

    std::sort(sort_buf.begin(), sort_buf.begin() + center_count, cv::hough_cmp_gt(adata));
    cvClearSeq( centers );
    cvSeqPushMulti( centers, &sort_buf[0], center_count );

    dist_buf = cvCreateMat( 1, nz_count, CV_32FC1 );
    ddata = dist_buf->data.fl;

    dr = dp;
    min_dist = MAX( min_dist, dp );
    min_dist *= min_dist;
    // For each found possible center
    // Estimate radius and check support
    for( i = 0; i < centers->total; i++ )
    {
        int ofs = *(int*)cvGetSeqElem( centers, i );
        y = ofs/(acols+2);
        x = ofs - (y)*(acols+2);
        //Calculate circle's center in pixels
        float cx = (float)((x + 0.5f)*dp), cy = (float)(( y + 0.5f )*dp);
        float start_dist, dist_sum;
        float r_best = 0;
        int max_count = 0;
        // Check distance with previously detected circles
        for( j = 0; j < circles->total; j++ )
        {
            float* c = (float*)cvGetSeqElem( circles, j );
            if( (c[0] - cx)*(c[0] - cx) + (c[1] - cy)*(c[1] - cy) < min_dist )
                break;
        }

        if( j < circles->total )
            continue;
        // Estimate best radius
        cvStartReadSeq( nz, &reader );
        for( j = k = 0; j < nz_count; j++ )
        {
            CvPoint pt;
            float _dx, _dy, _r2;
            CV_READ_SEQ_ELEM( pt, reader );
            _dx = cx - pt.x; _dy = cy - pt.y;
            _r2 = _dx*_dx + _dy*_dy;
            if(min_radius2 <= _r2 && _r2 <= max_radius2 )
            {
                ddata[k] = _r2;
                sort_buf[k] = k;
                k++;
            }
        }

        int nz_count1 = k, start_idx = nz_count1 - 1;
        if( nz_count1 == 0 )
            continue;
        dist_buf->cols = nz_count1;
        cvPow( dist_buf, dist_buf, 0.5 );
        std::sort(sort_buf.begin(), sort_buf.begin() + nz_count1, cv::hough_cmp_gt((int*)ddata));

        dist_sum = start_dist = ddata[sort_buf[nz_count1-1]];
        for( j = nz_count1 - 2; j >= 0; j-- )
        {
            float d = ddata[sort_buf[j]];

            if( d > max_radius )
                break;

            if( d - start_dist > dr )
            {
                float r_cur = ddata[sort_buf[(j + start_idx)/2]];
                if( (start_idx - j)*r_best >= max_count*r_cur ||
                    (r_best < FLT_EPSILON && start_idx - j >= max_count) )
                {
                    r_best = r_cur;
                    max_count = start_idx - j;
                }
                start_dist = d;
                start_idx = j;
                dist_sum = 0;
            }
            dist_sum += d;
        }
        // Check if the circle has enough support
        if( max_count > acc_threshold )
        {
            float c[3];
            c[0] = cx;
            c[1] = cy;
            c[2] = (float)r_best;
            cvSeqPush( circles, c );
            if( circles->total > circles_max )
                return;
        }
    }
}

CV_IMPL CvSeq*
cvHoughCircles( CvArr* src_image, void* circle_storage,
                int method, double dp, double min_dist,
                double param1, double param2,
                int min_radius, int max_radius )
{
    CvSeq* result = 0;

    CvMat stub, *img = (CvMat*)src_image;
    CvMat* mat = 0;
    CvSeq* circles = 0;
    CvSeq circles_header;
    CvSeqBlock circles_block;
    int circles_max = INT_MAX;
    int canny_threshold = cvRound(param1);
    int acc_threshold = cvRound(param2);

    img = cvGetMat( img, &stub );

    if( !CV_IS_MASK_ARR(img))
        CV_Error( CV_StsBadArg, "The source image must be 8-bit, single-channel" );

    if( !circle_storage )
        CV_Error( CV_StsNullPtr, "NULL destination" );

    if( dp <= 0 || min_dist <= 0 || canny_threshold <= 0 || acc_threshold <= 0 )
        CV_Error( CV_StsOutOfRange, "dp, min_dist, canny_threshold and acc_threshold must be all positive numbers" );

    min_radius = MAX( min_radius, 0 );
    if( max_radius <= 0 )
        max_radius = MAX( img->rows, img->cols );
    else if( max_radius <= min_radius )
        max_radius = min_radius + 2;

    if( CV_IS_STORAGE( circle_storage ))
    {
        circles = cvCreateSeq( CV_32FC3, sizeof(CvSeq),
            sizeof(float)*3, (CvMemStorage*)circle_storage );
    }
    else if( CV_IS_MAT( circle_storage ))
    {
        mat = (CvMat*)circle_storage;

        if( !CV_IS_MAT_CONT( mat->type ) || (mat->rows != 1 && mat->cols != 1) ||
            CV_MAT_TYPE(mat->type) != CV_32FC3 )
            CV_Error( CV_StsBadArg,
            "The destination matrix should be continuous and have a single row or a single column" );

        circles = cvMakeSeqHeaderForArray( CV_32FC3, sizeof(CvSeq), sizeof(float)*3,
                mat->data.ptr, mat->rows + mat->cols - 1, &circles_header, &circles_block );
        circles_max = circles->total;
        cvClearSeq( circles );
    }
    else
        CV_Error( CV_StsBadArg, "Destination is not CvMemStorage* nor CvMat*" );

    switch( method )
    {
    case CV_HOUGH_GRADIENT:
        icvHoughCirclesGradient( img, (float)dp, (float)min_dist,
                                min_radius, max_radius, canny_threshold,
                                acc_threshold, circles, circles_max );
          break;
    default:
        CV_Error( CV_StsBadArg, "Unrecognized method id" );
    }

    if( mat )
    {
        if( mat->cols > mat->rows )
            mat->cols = circles->total;
        else
            mat->rows = circles->total;
    }
    else
        result = circles;

    return result;
}


namespace cv
{

const int STORAGE_SIZE = 1 << 12;

static void seqToMat(const CvSeq* seq, OutputArray _arr)
{
    if( seq && seq->total > 0 )
    {
        _arr.create(1, seq->total, seq->flags, -1, true);
        Mat arr = _arr.getMat();
        cvCvtSeqToArray(seq, arr.data);
    }
    else
        _arr.release();
}

}

void cv::HoughCircles( InputArray _image, OutputArray _circles,
                       int method, double dp, double min_dist,
                       double param1, double param2,
                       int minRadius, int maxRadius )
{
    Ptr<CvMemStorage> storage = cvCreateMemStorage(STORAGE_SIZE);
    Mat image = _image.getMat();
    CvMat c_image = image;
    CvSeq* seq = cvHoughCircles( &c_image, storage, method,
                    dp, min_dist, param1, param2, minRadius, maxRadius );
    seqToMat(seq, _circles);
}

/* End of file. */
