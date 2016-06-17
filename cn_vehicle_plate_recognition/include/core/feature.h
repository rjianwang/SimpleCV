/* \file feature.h
 * Compute features of characters
 */

#include "../stdafx.h"
#include "char.h"

/* \namespace pr
 * Namespace where all C++ Plate Recognition functionality reside
 */
namespace pr
{
    
    cv::Mat features(const cv::Mat &charImg);
    std::vector<float> gradientF(const cv::Mat &img);
    float sumMat(const cv::Mat &img);
    cv::Mat pixelF(const cv::Mat &img);
    cv::Mat hhistF(const cv::Mat &img);
    cv::Mat vhistF(const cv::Mat &img);

} /* end for namespace pr */
