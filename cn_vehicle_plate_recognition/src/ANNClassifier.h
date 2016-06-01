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
    void train(const int num_classes);
    int predict(const cv::Mat &sample);

    void load_xml(const std::string filename);
    void load_cn_data(const std::string filepath);
    void load_data(const std::string filepath);

private:
    CvANN_MLP ann;
    cv::Mat trainData;
    cv::Mat labelData;

    const int numCharacters; // = 36

public:
    bool DEBUG;
};
