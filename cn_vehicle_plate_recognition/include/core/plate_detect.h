/* \file plate_detection.h
 * Definition of class PlateDetection
 */
#pragma once

#include "../stdafx.h"
#include <vector>

/* \namespace pr
 * Namespace where all C++ Plate Recognition functionality resides
 */
namespace pr
{

class Plate;

/* \class PlateDetection
 * Definition of plate detect methods
 */
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
    
}; /*end for class PlateDetection */

} /* end for namespace pr */
