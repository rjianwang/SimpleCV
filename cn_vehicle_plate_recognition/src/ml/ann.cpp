/* \file ann.cpp
 * Ann implementation of ANN classifier
 */
#include "../../include/core/resource.h"
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

    params.train_method = CvANN_MLP_TrainParams::BACKPROP;
    params.bp_dw_scale = 0.1;
    params.bp_moment_scale = 0.1;
}

ANNClassifier::~ANNClassifier()
{

}

void ANNClassifier::load_xml(const std::string filename)
{
    if (DEBUG_MODE)
        std::cout << "Loading model for ANN classifier." << std::endl;

    cv::FileStorage fs("OCR.xml", cv::FileStorage::READ);
    fs["TrainingDataF15"] >> trainData;
    fs["classes"] >> labelData;
    
    fs.release();
}

void ANNClassifier::load_cn_data(const std::string filepath)
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
            
            // resize to 20 * 20 
            cv::Mat resized;
            resized.create(20, 20, CV_32FC1);
            resize(img, resized, resized.size(), 0, 0, cv::INTER_CUBIC);

            cv::Mat f = features(resized, 15);
            f = f.reshape(1, 1);
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
            
            // resize to 20 * 20
            cv::Mat resized;
            resized.create(20, 20, CV_32FC1);
            resize(img, resized, resized.size(), 0, 0, cv::INTER_CUBIC);
            
            cv::Mat f = features(resized, 15);

            f = f.reshape(1, 1);
            trainData.push_back(f);
            labelData.push_back(n);
        }
    }
}

cv::Mat ANNClassifier::ProjectedHistogram(cv::Mat img, int t)
{
	int sz = (t) ? img.rows : img.cols;
    cv::Mat mhist = cv::Mat::zeros(1, sz, CV_32F);

	for (int j = 0; j<sz; j++){
        cv::Mat data = (t) ? img.row(j) : img.col(j);
		mhist.at<float>(j) = cv::countNonZero(data);
	}

	// ¹éÒ»»¯Ö±·œÍŒ
	double min, max;
    cv::minMaxLoc(mhist, &min, &max);

	if (max>0)
		mhist.convertTo(mhist, -1, 1.0f / max, 0);

	return mhist;
}

cv::Mat ANNClassifier::getVisualHistogram(cv::Mat *hist, int type)
{
	int size = 100;
    cv::Mat imHist;

	if (type == HORIZONTAL){
		imHist.create(cv::Size(size, hist->cols), CV_8UC3);
	}
	else{
		imHist.create(cv::Size(hist->cols, size), CV_8UC3);
	}

	imHist = cv::Scalar(55, 55, 55);

	for (int i = 0; i<hist->cols; i++){
		float value = hist->at<float>(i);
		int maxval = (int)(value*size);

        cv::Point pt1;
        cv::Point pt2, pt3, pt4;

		if (type == HORIZONTAL)
		{
			pt1.x = pt3.x = 0;
			pt2.x = pt4.x = maxval;
			pt1.y = pt2.y = i;
			pt3.y = pt4.y = i + 1;

            cv::line(imHist, pt1, pt2, CV_RGB(220, 220, 220), 1, 8, 0);
            cv::line(imHist, pt3, pt4, CV_RGB(34, 34, 34), 1, 8, 0);

			pt3.y = pt4.y = i + 2;
            cv::line(imHist, pt3, pt4, CV_RGB(44, 44, 44), 1, 8, 0);
			pt3.y = pt4.y = i + 3;
            cv::line(imHist, pt3, pt4, CV_RGB(50, 50, 50), 1, 8, 0);
		}
		else
		{
			pt1.x = pt2.x = i;
			pt3.x = pt4.x = i + 1;
			pt1.y = pt3.y = 100;
			pt2.y = pt4.y = 100 - maxval;

            cv::line(imHist, pt1, pt2, CV_RGB(220, 220, 220), 1, 8, 0);
            cv::line(imHist, pt3, pt4, CV_RGB(34, 34, 34), 1, 8, 0);

			pt3.x = pt4.x = i + 2;
            cv::line(imHist, pt3, pt4, CV_RGB(44, 44, 44), 1, 8, 0);
			pt3.x = pt4.x = i + 3;
            cv::line(imHist, pt3, pt4, CV_RGB(50, 50, 50), 1, 8, 0);
		}
	}
	return imHist;
}

void ANNClassifier::drawVisualFeatures(cv::Mat character, cv::Mat hhist, 
        cv::Mat vhist, cv::Mat lowData){
    cv::Mat img(121, 121, CV_8UC3, cv::Scalar(0, 0, 0));
    cv::Mat ch;
    cv::Mat ld;

    cv::cvtColor(character, ch, CV_GRAY2RGB);

    cv::resize(lowData, ld, cv::Size(100, 100), 0, 0, cv::INTER_NEAREST);
    cv::cvtColor(ld, ld, CV_GRAY2RGB);

    cv::Mat hh = getVisualHistogram(&hhist, HORIZONTAL);
    cv::Mat hv = getVisualHistogram(&vhist, VERTICAL);

    cv::Mat subImg = img(cv::Rect(0, 101, 20, 20));
	ch.copyTo(subImg);

	subImg = img(cv::Rect(21, 101, 100, 20));
	hh.copyTo(subImg);

	subImg = img(cv::Rect(0, 0, 20, 100));
	hv.copyTo(subImg);

	subImg = img(cv::Rect(21, 0, 100, 100));
	ld.copyTo(subImg);

    cv::line(img, cv::Point(0, 100), cv::Point(121, 100), 
            cv::Scalar(0, 0, 255));
    cv::line(img, cv::Point(20, 0), cv::Point(20, 121), 
            cv::Scalar(0, 0, 255));

    cv::imshow("Visual Features", img);

    cv::waitKey(0);
}

// ÌØÕ÷ÌáÈ¡
cv::Mat ANNClassifier::features(cv::Mat in, int sizeData){
	//Histogram features
    cv::Mat vhist = ProjectedHistogram(in, VERTICAL);
    cv::Mat hhist = ProjectedHistogram(in, HORIZONTAL);

    cv::Mat lowData;
    cv::resize(in, lowData, cv::Size(sizeData, sizeData));

	/*if (DEBUG_MODE)
		drawVisualFeatures(in, hhist, vhist, lowData);*/

	int numCols = vhist.cols + hhist.cols + lowData.cols * lowData.cols;

    cv::Mat out = cv::Mat::zeros(1, numCols, CV_32F);

	int j = 0;
	for (int i = 0; i < vhist.cols; i++)
	{
		out.at<float>(j) = vhist.at<float>(i);
		j++;
	}
	for (int i = 0; i < hhist.cols; i++)
	{
		out.at<float>(j) = hhist.at<float>(i);
		j++;
	}
	for (int x = 0; x < lowData.cols; x++)
	{
		for (int y = 0; y < lowData.rows; y++){
			out.at<float>(j) = (float)lowData.at<unsigned char>(x, y);
			j++;
		}
	}
	return out;
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

	cv::Mat f = features(sample, 15);
    cv::Mat output(1, this->num_output, CV_32FC1);
    ann.predict(f, output);
    
    cv::Point maxLoc;
    double maxVal;
    cv::minMaxLoc(output, 0, &maxVal, 0, &maxLoc);

    return maxLoc.x;
}

} /* end for namespace pr */
