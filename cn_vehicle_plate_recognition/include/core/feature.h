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
    
    cv::Mat feature(const cv::Mat &charImg);
    std::vector<float> gradientF(const cv::Mat &img);
    float sumValue(const cv::Mat &img);

} /* end for namespace pr */
