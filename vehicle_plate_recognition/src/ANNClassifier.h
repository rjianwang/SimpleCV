#pragma once

#include "StdAfx.h"
#include <string>
#include <opencv2/ml/ml.hpp>

class ANNClassifier
{
public:
    ANNClassifier();
    ~ANNClassifier();

public:
    void train(const cv::Mat &trainData, const cv::Mat &labelData, 
            int nlayers = 10);
    int predict(const cv::Mat &sample);

    cv::FileStorage load_xml(std::string filename);

private:
    CvANN_MLP ann;

    const int numCharacters; // = 30
};
