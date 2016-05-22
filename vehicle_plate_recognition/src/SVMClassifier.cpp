#include "SVMClassifier.h"

SVMClassifier::SVMClassifier()
{
    SVM_params.svm_type = CvSVM::C_SVC;
    SVM_params.kernel_type = CvSVM::LINEAR; // CvSVM::LINEAR
    SVM_params.degree = 0;
    SVM_params.gamma = 1; 
    SVM_params.coef0 = 0; 
    SVM_params.C = 1;
    SVM_params.nu = 0;    
    SVM_params.p = 0;     
    SVM_params.term_crit = cvTermCriteria(CV_TERMCRIT_ITER, 1000, 0.01);

    svmClassifier = NULL;
}

SVMClassifier::~SVMClassifier()
{
    if (svmClassifier != NULL)
        delete svmClassifier;
}

bool SVMClassifier::train(const cv::Mat &trainData, const cv::Mat &labelData)
{
    svmClassifier = new CvSVM(trainData, labelData, cv::Mat(), cv::Mat(), SVM_params);
}

float SVMClassifier::predict(const cv::Mat &sample)
{
    float response = svmClassifier->predict(sample);
    return response;
}
