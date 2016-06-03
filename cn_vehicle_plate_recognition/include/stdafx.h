#pragma once

#include <iostream>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv/cvaux.h>

/* \namespace pr
 *  Namespace where all C++ Plate Recognition functionality resides
 */
namespace pr
{
    extern bool DEBUG_MODE;     // allow or not to output debug info
    extern bool DETECT_MODE;    // enable or not to detect plate only
} /* end for namespace pr */
