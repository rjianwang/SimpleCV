#pragma once

#include "StdAfx.h"
#include <vector>
#include "Plate.h"

class Plate;

class PlateDetection
{
public:
	PlateDetection();
	~PlateDetection();

public:
	cv::Mat histeq(cv::Mat img);
	bool verifySizes(cv::RotatedRect ROI);
	std::vector<Plate> segment(cv::Mat img);

public:
	std::string filename;
	bool saveRecognition;
    
    bool DEBUG;
};

