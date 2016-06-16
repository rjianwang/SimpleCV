#include <string>
#include "../../include/core/ocr2.h"
#include "../../include/core/resource.h"
#include "../../include/core/feature.h"
#include "../../include/tool/tool.h"

using namespace cv;

namespace pr
{
    Mat trainData;
    Mat labelData;

    void load_cn(const std::string filepath)
    {
        if (DEBUG_MODE)
        {
            std::cout << "Loading training data(Chinese Characters) for SVM classifier." 
                << std::endl;
        }
        if (!trainData.empty())
        {
            trainData.release();
            labelData.release();
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

                cv::Mat resized;
                resized.create(23, 13, CV_32FC1);
                cv::resize(img, resized, resized.size());
                resized = resized.reshape(1, 1);

                trainData.push_back(resized);
                labelData.push_back(n);
            }
        }
    }

    void load_data(const std::string filepath)
    {
        if (DEBUG_MODE)
        {
            std::cout << "Loading training data(digits & letters) for SVM classifier." 
                << std::endl;
        }
        if (!trainData.empty())
        {
            trainData.release();
            labelData.release();
        }

        for (int n = 0; n < 34; n++)
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

                cv::Mat resized;
                resized.create(23, 13, CV_32FC1);
                cv::resize(img, resized, resized.size());
                resized = resized.reshape(1, 1);

                trainData.push_back(resized);
                labelData.push_back(n);
            }
        }
    }

    std::vector<int> ocr1()
    {
        load_cn("../data/cn_chars/");

        // Set up SVM's parameters
        CvSVMParams params;
        params.svm_type    = CvSVM::C_SVC;
        params.C           = 0.1; 
        params.kernel_type = CvSVM::LINEAR;
        params.term_crit   = cvTermCriteria(CV_TERMCRIT_ITER, 1000, 1e-6);

        // Train the SVM
        CvSVM SVM;
        trainData.convertTo(trainData, CV_32FC1);
        SVM.train(trainData, labelData, Mat(), Mat(), params);

        if (DEBUG_MODE)
            std::cout << "Predicting..." << std::endl;
        std::vector<int> ret;
        cv::Mat img = cv::imread("PlateNumber0.jpg", CV_LOAD_IMAGE_GRAYSCALE);
        cv::Mat resized;
        resized.create(23, 13, CV_32FC1);
        cv::resize(img, resized, resized.size());
        resized = resized.reshape(1, 1);
        resized.convertTo(resized, CV_32FC1);

        float response = SVM.predict(resized);
        ret.push_back(int(response));

        waitKey(0);

        return ret;
    }

    std::vector<int> ocr2()
    {
        load_data("../data/charSamples/");

        // Set up SVM's parameters
        CvSVMParams params;
        params.svm_type    = CvSVM::C_SVC;
        params.C           = 0.1; 
        params.kernel_type = CvSVM::LINEAR;
        params.term_crit   = cvTermCriteria(CV_TERMCRIT_ITER, 1000, 1e-6);

        // Train the SVM
        CvSVM SVM;
        trainData.convertTo(trainData, CV_32FC1);
        SVM.train(trainData, labelData, Mat(), Mat(), params);

        if (DEBUG_MODE)
            std::cout << "Predicting..." << std::endl;
        std::vector<int> ret;
        for (int i = 1; i <= 6; i++)
        {
            std::stringstream ss;
            ss << "PlateNumber" << i << ".jpg";
            cv::Mat img = cv::imread(ss.str().c_str(), CV_LOAD_IMAGE_GRAYSCALE);
            cv::Mat resized;
            resized.create(23, 13, CV_32FC1);
            cv::resize(img, resized, resized.size());
            resized = resized.reshape(1, 1);
            resized.convertTo(resized, CV_32FC1);

            float response = SVM.predict(resized);
            ret.push_back(int(response));
        }

        return ret;
    }
}
