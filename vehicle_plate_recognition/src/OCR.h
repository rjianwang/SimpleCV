#pragma once

#include "StdAfx.h"
#include <string.h>
#include <vector>

#include "Plate.h"
#include <opencv2/ml/ml.hpp>

class ANNClassifier;

#define HORIZONTAL    1
#define VERTICAL    0

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

    cv::Mat features(cv::Mat input, int size);
    std::string ocr(Plate *input);
	int charSize;
    cv::Mat preprocessChar(cv::Mat in);

	int classifyKnn(cv::Mat f);
	void trainKnn(cv::Mat trainSamples, cv::Mat trainClasses, int k);

private:
	bool trained;
    std::vector<CharSegment> segment(Plate input);
    cv::Mat Preprocess(cv::Mat in, int newSize);
    cv::Mat getVisualHistogram(cv::Mat *hist, int type);
	void drawVisualFeatures(cv::Mat character, cv::Mat hhist, 
            cv::Mat vhist, cv::Mat lowData);
    cv::Mat ProjectedHistogram(cv::Mat img, int t);
	bool verifySizes(cv::Mat r);

    ANNClassifier *annClassifier;
};
