/* \file ann.cpp
 * Ann implementation of ANN classifier
 */
#include "../../include/core/resource.h"
#include "../../include/core/feature.h"
#include "../../include/ml/ann.h"
#include "../../include/tool/tool.h"

/* \namespace pr
 * Namespace where all C++ Plate Recognition functionality resides
 */
namespace pr
{

/* \class ANNClassifier
 * A customized ANN classifier
 */
ANNClassifier::ANNClassifier(int num_neurons, int num_output)
{
    this->num_neurons = num_neurons;
    this->num_output = num_output;

/*    params.train_method = CvANN_MLP_TrainParams::BACKPROP;
    params.bp_dw_scale = 0.1;
    params.bp_moment_scale = 0.1;
    */
}

ANNClassifier::~ANNClassifier()
{

}

void ANNClassifier::load(const std::string filename)
{
    if (DEBUG_MODE)
        std::cout << "Loading model for ANN classifier." << std::endl;
    ann.load(filename.c_str());
}

void ANNClassifier::load_cn(const std::string filepath)
{
    if (DEBUG_MODE)
    {
        std::cout << "Loading training data(Chinese Characters) for ANN classifier." 
            << std::endl;
    }

    for (int n = 0; n < Resources::numCNCharacters; n++)
    {
        std::vector<std::string> files;
        files = getFiles(filepath + Resources::cn_chars[n] + "/");

        if (files.size() == 0)
            std::cout << "Loading trainning data(Chinese Characters) ERROR. "
                << "Directory \"" << filepath + Resources::cn_chars[n] << "\" is empty." 
                << std::endl;

        for (int i = 0; i < files.size(); i++)
        {
            std::string path = filepath + Resources::cn_chars[n] + "/" + files[i];
            cv::Mat img = cv::imread(path.c_str(), CV_LOAD_IMAGE_GRAYSCALE);

            if (img.cols == 0)
                std::cout << "Fail to load images " << path << std::endl;
            
            cv::threshold(img, img, 0, 255, CV_THRESH_OTSU + CV_THRESH_BINARY);

            cv::Mat f = features(img);
            trainData.push_back(f);
            labelData.push_back(n);
        }
    }
}

void ANNClassifier::load_data(const std::string filepath)
{
    if (DEBUG_MODE)
    {
        std::cout << "Loading training data(digits & letters) for ANN classifier." 
            << std::endl;
    }

    for (int n = 0; n < Resources::numCharacters; n++)
    {
        std::vector<std::string> files;
        files = getFiles(filepath + Resources::chars[n] + "/");

        if (files.size() == 0)
            std::cout << "Loading trainning data ERROR. " 
                << "Directory \"" << filepath + Resources::chars[n] << "\" is empty."
                << std::endl;

        for (int i = 0; i < files.size(); i++)
        {
            std::string path = filepath + Resources::chars[n] + "/" + files[i];
            cv::Mat img = cv::imread(path.c_str(), CV_LOAD_IMAGE_GRAYSCALE);

            if (img.cols == 0)
                std::cout << "Fail to load images " << path << std::endl;
            
            cv::threshold(img, img, 0, 255, CV_THRESH_OTSU + CV_THRESH_BINARY);
            
            cv::Mat f = features(img);

            trainData.push_back(f);
            labelData.push_back(n);
        }
    }
}

void ANNClassifier::train()
{
    if (DEBUG_MODE)
    {
        std::cout << "Training..." << std::endl;
        std::cout << "\ttrainData size: " << trainData.size() << std::endl;
        std::cout << "\tlabelData size: " << labelData.size() << std::endl;
    } 
    cv::Mat layers(1, 3, CV_32SC1);
    layers.at<int>(0) = trainData.cols;
    layers.at<int>(1) = this->num_neurons;
    layers.at<int>(2) = this->num_output;;
    ann.create(layers, CvANN_MLP::SIGMOID_SYM, 1, 1);

    cv::Mat trainClasses;
    trainClasses.create(trainData.rows, this->num_output, CV_32FC1);
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

    trainData.convertTo(trainData, CV_32FC1);
    ann.train(trainData, trainClasses, weights);
}

int ANNClassifier::predict(const cv::Mat &sample)
{
    if (DEBUG_MODE)
    {
        std::cout << "Predicting..." << std::endl;
        std::cout << "\tSample size: " << sample.size() << std::endl;
    }

    std::cout << sample << std::endl;
    cv::Mat output(1, this->num_output, CV_32FC1);
    ann.predict(sample, output);
    
    cv::Point maxLoc;
    double maxVal;
    cv::minMaxLoc(output, 0, &maxVal, 0, &maxLoc);

    return maxLoc.x;
}

} /* end for namespace pr */
