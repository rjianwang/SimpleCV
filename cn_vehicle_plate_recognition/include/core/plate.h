#pragma once

#include "../stdafx.h"

#include <string.h>
#include <vector>

class Plate
{
public:
	Plate();
	Plate(cv::Mat img, cv::Rect pos);

    std::string str();

public:
    cv::Rect position;
    cv::Mat image;

    std::vector<std::string> chars;
    std::vector<cv::Rect> charsPos;
};
