#pragma once

#include "StdAfx.h"

#include <string.h>
#include <vector>

class Plate{
public:
	Plate();
	Plate(cv::Mat img, cv::Rect pos);
    std::string str();
    cv::Rect position;
    cv::Mat image;
    std::vector<char> chars;
    std::vector<cv::Rect> charsPos;
};
