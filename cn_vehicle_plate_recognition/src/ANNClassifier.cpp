#include "ANNClassifier.h"
#include "Util.h"
#include "Resources.h"

ANNClassifier::ANNClassifier(): numCharacters(36)
{
    DEBUG = false;
}

ANNClassifier::~ANNClassifier()
{

}

void ANNClassifier::load_xml(const std::string filename)
{
    if (DEBUG)
        std::cout << "Loading model for ANN classifier." << std::endl;

    cv::FileStorage fs("OCR.xml", cv::FileStorage::READ);
    fs["TrainingDataF15"] >> trainData;
    fs["classes"] >> labelData;
    
    fs.release();
}

void ANNClassifier::load_cn_data(const std::string filepath)
{
    if (DEBUG)
        std::cout << "Loading training data(Chinese Characters) for ANN classifier." 
            << std::endl;

    std::vector<std::string> labels;
    labels = Util::getFiles(filepath);
    for (int n = 0; n < labels.size(); n++)
    {
        std::vector<std::string> files;
        files = Util::getFiles(filepath + labels[n] + "/");

        if (files.size() == 0)
            std::cout << "Loading trainning data(Chinese Characters) ERROR. "
                << "Directory \"" << filepath + labels[n] << "\" is empty." 
                << std::endl;

        for (int i = 0; i < files.size(); i++)
        {
            std::string path = filepath + labels[n] + "/" + files[i];
            cv::Mat img = cv::imread(path.c_str(), CV_LOAD_IMAGE_GRAYSCALE);

            if (img.cols == 0)
                std::cout << "Fail to load images " << path << std::endl;
            
            // resize to 20 * 20 
            cv::Mat resized;
            resized.create(20, 20, CV_32FC1);
            resize(img, resized, resized.size(), 0, 0, cv::INTER_CUBIC);

            resized = resized.reshape(1, 1);
            trainData.push_back(resized);
            labelData.push_back(n);
        }
    }
}

void ANNClassifier::load_data(const std::string filepath)
{
    if (DEBUG)
        std::cout << "Loading training data for ANN classifier." 
            << std::endl;

    std::vector<std::string> labels;
    labels = Util::getFiles(filepath);
    for (int n = 0; n < labels.size(); n++)
    {
        std::vector<std::string> files;
        files = Util::getFiles(filepath + labels[n] + "/");

        if (files.size() == 0)
            std::cout << "Loading trainning data ERROR. " 
                << "Directory \"" << filepath + labels[n] << "\" is empty."
                << std::endl;

        for (int i = 0; i < files.size(); i++)
        {
            std::string path = filepath + labels[n] + "/" + files[i];
            cv::Mat img = cv::imread(path.c_str(), CV_LOAD_IMAGE_GRAYSCALE);

            if (img.cols == 0)
                std::cout << "Fail to load images " << path << std::endl;
            
            // resize to 12 * 24 
            cv::Mat resized;
            resized.create(12, 24, CV_32FC1);
            resize(img, resized, resized.size(), 0, 0, cv::INTER_CUBIC);

            resized = resized.reshape(1, 1);
            trainData.push_back(resized);
            labelData.push_back(n);
        }
    }
}

void ANNClassifier::train(const int num_classes)
{
    if (DEBUG)
    {
        std::cout << "Training..." << std::endl;
        std::cout << "\ttrainData size: " << trainData.size() << std::endl;
        std::cout << "\tlabelData size: " << labelData.size() << std::endl;
    } 
    cv::Mat layers(1, 3, CV_32SC1);
    layers.at<int>(0) = trainData.cols;
    layers.at<int>(1) = 10;
    layers.at<int>(2) = num_classes;
    ann.create(layers, CvANN_MLP::SIGMOID_SYM, 1, 1);

    cv::Mat trainClasses;
    trainClasses.create(trainData.rows, num_classes, CV_32FC1);
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
    if (DEBUG)
    {
        std::cout << "Predicting..." << std::endl;
        std::cout << "\tSample size: " << sample.size() << std::endl;
    }

    cv::Mat output(1, numCharacters, CV_32FC1);
    ann.predict(sample, output);
    
    cv::Point maxLoc;
    double maxVal;
    cv::minMaxLoc(output, 0, &maxVal, 0, &maxLoc);

    return maxLoc.x;
}
