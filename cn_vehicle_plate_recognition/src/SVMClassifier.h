#include "StdAfx.h"
#include <opencv2/ml/ml.hpp>

class SVMClassifier
{
public:
    SVMClassifier();
    ~SVMClassifier();

public:
    void load_data(std::string filepath);
    void load_model(std::string filename);

public:
    bool train();
    void save(std::string filepath);
    float predict(const cv::Mat &sample);

private:
    cv::Mat imresize(int height, int width);

public:
    bool DEBUG;

private:
    CvSVMParams SVM_params;
    CvSVM *svmClassifier;

    cv::Mat trainData;
    cv::Mat labelData;
};
