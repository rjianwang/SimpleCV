#pragma once

#include "../stdafx.h"
#include <vector>
#include "plate.h"

class Plate;

class PlateDetection
{
public:
	PlateDetection();
	~PlateDetection();

public:
	cv::Mat histeq(cv::Mat img);
	bool verifySizes(cv::RotatedRect ROI);
	bool verifySizes(cv::Rect ROI);
	std::vector<Plate> segment(cv::Mat img);

public:
	std::string filename;
	bool saveRecognition;
    
    bool DEBUG;
};

