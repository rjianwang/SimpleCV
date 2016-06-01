#include "SVMClassifier.h"
#include "Util.h"

SVMClassifier::SVMClassifier()
{
    SVM_params.svm_type = CvSVM::C_SVC;
    SVM_params.kernel_type = CvSVM::LINEAR; // CvSVM::LINEAR
    SVM_params.degree = 0; // 0
    SVM_params.gamma = 1;  // 1
    SVM_params.coef0 = 0;  // 0
    SVM_params.C = 1;
    SVM_params.nu = 0;     // 0
    SVM_params.p = 0;      // 0
    SVM_params.term_crit = cvTermCriteria(CV_TERMCRIT_ITER, 10000, 0.0001);

    svmClassifier = NULL;

    DEBUG = false;
}

SVMClassifier::~SVMClassifier()
{
    if (svmClassifier != NULL)
        delete svmClassifier;
}

// Load training datas from directory
// For EXAMPLE: 
//     There is a directory "train" and two sub-directories in it,
//     that is "plates0", "plates1". 
//     We load datas from "train".
void SVMClassifier::load_data(std::string filepath)
{
    if (DEBUG)
        std::cout << "Loading training data for SVM classifier." << std::endl;

    std::string paths[2] = {filepath + "/plates_0/", filepath + "plates_1/"};
    for (int n = 0; n < 2; n++)
    {
        std::vector<std::string> files;
        files = Util::getFiles(paths[n]);

        if (files.size() == 0)
            std::cout << "Load trainging data ERROR: empty directory" << std::endl;

        for (int i = 0; i < files.size(); i++)
        {
            std::string path = paths[n] + files[i];
            cv::Mat img = cv::imread(path.c_str());
            cv::cvtColor(img, img, CV_BGR2GRAY);

            if (img.cols == 0)
               std::cout << "Fail to load images " << files[i] << std::endl;

            // resized to 33 * 144
            cv::Mat resized;
            resized.create(33, 144, CV_32FC1);
            resize(img, resized, resized.size(), 0, 0, cv::INTER_CUBIC);

            resized = resized.reshape(1, 1);
            trainData.push_back(resized);
            labelData.push_back(n);
        }
    }
}

// Load training model
void SVMClassifier::load_model(std::string filename)
{
    if (DEBUG)
        std::cout << "Loading model for SVM classifier." << std::endl;

    cv::FileStorage fs(filename, cv::FileStorage::READ);
    fs["TrainingData"] >> trainData;
    fs["classes"] >> labelData;

    fs.release();
}

// train
bool SVMClassifier::train()
{
    if (DEBUG)
    {
        std::cout << "Training ..." << std::endl;
        std::cout << "\ttrainData size: " << trainData.size() << std::endl;
        std::cout << "\tlabelData size: " << labelData.size() << std::endl;
    }
    
    trainData.convertTo(trainData, CV_32FC1);
    svmClassifier = new CvSVM(trainData, labelData, cv::Mat(), cv::Mat(), SVM_params);
}

void SVMClassifier::save(std::string filepath)
{
    if (DEBUG)
    {
        std::cout << "Saving training model..." << std::endl;
        std::cout << "\tSVM_cn.xml" << std::endl;
    }

    cv::FileStorage fs(filepath.c_str(), cv::FileStorage::WRITE);
    svmClassifier->write(*fs, "svm");
}

// predict
float SVMClassifier::predict(const cv::Mat &sample)
{
    if (DEBUG)
    {
        std::cout << "Predicting..." << std::endl;
        std::cout << "\tSample size: " << sample.size() << std::endl;
    }

    float response = svmClassifier->predict(sample);
    return response;
}
