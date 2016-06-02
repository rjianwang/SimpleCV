#pragma once

#include "../stdafx.h"
#include <string.h>
#include <vector>

#include <opencv2/ml/ml.hpp>

class ANNClassifier;
class Plate;

class CharSegment{
public:
	CharSegment();
	CharSegment(cv::Mat i, cv::Rect p);
    cv::Mat img;
    cv::Rect pos;
};

class OCR{
public:
    OCR();
    ~OCR();

public:
	bool DEBUG;
	bool saveSegments;
    std::string filename;
	static const int numCharacters;
	static const char strCharacters[];

    std::string ocr(Plate *input);
	int charSize;
    cv::Mat preprocessChar(cv::Mat in);

private:
	bool trained;
    std::vector<CharSegment> segment(Plate input);
    int getSpecificChar(const Plate &plate, const std::vector<CharSegment> &input);
    CharSegment getChineseChar(const cv::Mat &img, const CharSegment &speck);
    cv::Mat Preprocess(cv::Mat in, int newSize);
    
	bool verifySizes(cv::Mat r);
    cv::Mat removeMD(cv::Mat img);

    void process_chars(Plate *input, 
            const std::vector<CharSegment> segments);
    void process_cn_chars(Plate *input, 
            const std::vector<CharSegment> segments);
};
