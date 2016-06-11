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
	bool verifySizes(const cv::RotatedRect &ROI);
	bool verifyNonZero(const cv::Mat &img, float thresh = 0.1);
    bool verifyColorJump(const cv::Mat &img, int threshold = 7);
    
    cv::Mat preprocessImg(cv::Mat &img);
    cv::Mat segment(const cv::Mat &img, const cv::RotatedRect &rc);
    std::vector<Plate> detectCore(const cv::Mat &img, const cv::Mat &thresh);
    std::vector< std::vector<int> > colorJump(cv::Mat &img);
    cv::Mat colorMatch(const cv::Mat &img, const char color,
            const bool adaptive_minsv = false);
    std::vector<Plate> colorDetect(cv::Mat &img);
	std::vector<Plate> detect(cv::Mat &img);
	std::vector<Plate> sobelDetect(cv::Mat &img);

public:
	std::string filename;
	bool saveRecognition;
    
}; /*end for class PlateDetection */

} /* end for namespace pr */
