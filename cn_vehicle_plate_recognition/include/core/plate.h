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
    std::vector<cv::Point> points; // 车牌的左上点和右下点
    cv::Mat image;

    std::vector<std::string> chars;
    std::vector<cv::Rect> charsPos;
};

} /* end for namespace pr */
