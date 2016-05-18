#include "ANNClassifier.h"

ANNClassifier::ANNClassifier(): numCharacters(30)
{

}

ANNClassifier::~ANNClassifier()
{

}

cv::FileStorage ANNClassifier::load_xml(const std::string filename)
{
    cv::FileStorage fs("OCR.xml", cv::FileStorage::READ);
    return fs;
}

void ANNClassifier::train(const cv::Mat &trainData, 
        const cv::Mat &labelData,
        int nlayers)
{
    cv::Mat layers(1, 3, CV_32SC1);
    layers.at<int>(0) = trainData.cols;
    layers.at<int>(1) = nlayers;
    layers.at<int>(2) = numCharacters;
    ann.create(layers, CvANN_MLP::SIGMOID_SYM, 1, 1);

    cv::Mat trainClasses;
    trainClasses.create(trainData.rows, numCharacters, CV_32FC1);
    for (int i = 0; i < trainClasses.rows; i++)
    {
        for (int k = 0; k < trainClasses.cols; k++)
        {
            if (k == labelData.at<int>(i))
                trainClasses.at<float>(i, k) = 1;
            else
                trainClasses.at<float>(i, k) = 0;
        }
    }

    cv::Mat weights(1, trainData.rows, CV_32FC1, cv::Scalar::all(1));

    ann.train(trainData, trainClasses, weights);
}

int ANNClassifier::predict(const cv::Mat &sample)
{
    cv::Mat output(1, numCharacters, CV_32FC1);
    ann.predict(sample, output);
    
    cv::Point maxLoc;
    double maxVal;
    cv::minMaxLoc(output, 0, &maxVal, 0, &maxLoc);

    return maxLoc.x;
}
