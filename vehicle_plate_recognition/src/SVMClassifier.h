#include "StdAfx.h"
#include <opencv2/ml/ml.hpp>

class SVMClassifier
{
public:
    SVMClassifier();
    ~SVMClassifier();

public:
    bool train(const cv::Mat &trainData, const cv::Mat &labelData);
    float predict(const cv::Mat &sample);

private:
    CvSVMParams SVM_params;
    CvSVM *svmClassifier;
};
