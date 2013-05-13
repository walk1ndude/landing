#include "opencv2/imgproc/imgproc.hpp"

#include <stdio.h>
#include <stdlib.h>

using namespace cv;

void HoughLinesGrad(InputArray _image, OutputArray _lines, double rho, double theta, int threshold, double minLineLength, double maxGap, int line_lenght_min2, int line_length_max2);
