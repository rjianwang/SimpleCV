/* file svm.h
 * Definition of SVM Classifier
 */

#include "../stdafx.h"
#include <opencv2/ml/ml.hpp>

/* \namespace pr
 * Namespace where all C++ Plate Recognition functionality resides
 */
namespace pr
{

/* \class SVMClassifier
 * A customized SVM Classifier
 */
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

private:
    CvSVMParams SVM_params;
    CvSVM *svmClassifier;

    cv::Mat trainData;
    cv::Mat labelData;

}; /* end for class SVMClassifier */

} /* end for namespace pr */
