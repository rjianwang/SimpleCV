/* \file ann.h
 * An implementation of ANN Classifier
 */

#pragma once

#include "../stdafx.h"
#include <string>
#include <opencv2/ml/ml.hpp>

#define HORIZONTAL  1
#define VERTICAL    0

/* \namespace pr
 * Namespace where all C++ Plate Recognition functionality resides
 */
namespace pr
{

/* \class ANNClassifier
 * An customized ANN classifier
 */
class ANNClassifier
{
public:
    ANNClassifier(int num_neurons, int num_output);
    ~ANNClassifier();

public:
    void train();
    int predict(const cv::Mat &sample);

    void load_xml(const std::string filename);
    void load_cn_data(const std::string filepath);
    void load_data(const std::string filepath);
    
private:
	cv::Mat getVisualHistogram(cv::Mat *hist, int type);
	void drawVisualFeatures(cv::Mat character, cv::Mat hhist, 
            cv::Mat vhist, cv::Mat lowData);
    cv::Mat ProjectedHistogram(cv::Mat img, int t);
    cv::Mat features(cv::Mat input, int size);

private:
    CvANN_MLP ann;
    CvANN_MLP_TrainParams params;
    cv::Mat trainData;
    cv::Mat labelData;

private:
    int num_neurons;
    int num_output;

}; /* ends for class ANNClassifier */ 

} /* ends for namespace pr */
