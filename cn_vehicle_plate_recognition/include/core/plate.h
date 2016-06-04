/* \file plate.h
 * The definition of Class Plate
 */

#pragma once

#include "../stdafx.h"

#include <string.h>
#include <vector>

/* \namespace pr
 * Namespace where all C++ Plate  Recognition functionality redises
 * */
namespace pr
{

/* \class Plate
 * Definition of Class Plate
 * */
class Plate
{
public:
	Plate();
	Plate(cv::Mat img, cv::Rect pos);

    std::string str();

public:
    cv::Rect position;
    int width;   // 车牌缩放后的大小，默认为136
    int height;  // 车牌缩放后的高度，默认为36
    cv::Mat image;

    std::vector<std::string> chars;
    std::vector<cv::Rect> charsPos;
};

} /* end for namespace pr */
