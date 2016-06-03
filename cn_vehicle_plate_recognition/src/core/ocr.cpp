/* \file ocr.cpp
*  Implementation of Plate Recognition
*/

#include <string>
#include "../../include/core/plate.h"
#include "../../include/core/resource.h"
#include "../../include/core/char.h"
#include "../../include/core/ocr.h"
#include "../../include/tool/tool.h"
#include "../../include/ml/ann.h"

/* \namespace pr
*  Namespace where all the C++ Plate Recognition functionality resides
*/
namespace pr
{

/* \class OCR
*  Implementation of Plate Recognition
*/
OCR::OCR()
{
}

OCR::~OCR()
{
}

void OCR::preprocessPlate(Plate &plate)
{
	if (DEBUG_MODE)
	{
		cv::imshow("Plate Image", plate.image);
	}

    cv::Mat threshold;
    cv::threshold(plate.image, threshold, 190, 255, CV_THRESH_BINARY);

    if (DEBUG_MODE)
        cv::imshow("Threshold plate", threshold);
        
    plate.image = threshold;
}

cv::Mat OCR::removeMD(cv::Mat img)
{
    if (DEBUG_MODE)
    {
        std::cout << "Remove mao ding..." << std::endl;
    }

    int line = 4;
    int threshold = 15;

    for (int i = 0; i < line; i++)
    {
        int whiteCount = 0;
        for (int j = 0; j < img.cols; j++)
        {
            if (img.at<char>(i, j) == 255)
                whiteCount++;
        }
        for (int j = 0; whiteCount < threshold && j < img.cols; j++)
        {
            img.at<char>(i, j) = 0;
        }
    }

    for (int i = img.rows - line; i < img.rows; i++)
    {
        int whiteCount = 0;
        for (int j = 0; j < img.cols; j++)
        {
            if (img.at<char>(i, j) == 255)
                whiteCount++;
        }
        for (int j = 0; whiteCount < threshold && j < img.cols; j++)
        {
            img.at<char>(i, j) = 0;
        }
    }
    return img;
}

void OCR::process_chars(Plate &input, const std::vector<Char> &segments)
{
    // ANN Classifier for digits and letters
    //  ANNClassifier *annClassifier = new ANNClassifier(10, Resources::numSPCharacters);
    ANNClassifier *annClassifier = new ANNClassifier(100, Resources::numCharacters);

    //annClassifier->load_xml("../OCR.xml");
    annClassifier->load_data("../data/charSamples/");
    annClassifier->train();

    for (int i = 1; i < segments.size(); i++){
        // 对每个字符进行预处理，使得对所有图像均有相同的大小 
        cv::Mat ch = preprocessChar(segments[i].image);

        // 对于每个部分进行分类
        ch.convertTo(ch, CV_32FC1);
        int character = annClassifier->predict(ch);
        input.chars.push_back(std::string(1, Resources::sp_chars[character]));
        input.charsPos.push_back(segments[i].position);
    }

    delete annClassifier;
}

void OCR::process_cn_char(Plate &input, const Char &cn_char)
{
    // ANN Classifier for Chinese Characters
    ANNClassifier *annClassifier = new ANNClassifier(17, Resources::numCNCharacters);
    annClassifier->load_cn_data("../data/cn_chars/"); 
    annClassifier->train();

    // 对字符进行预处理
    cv::Mat ch = preprocessChar(cn_char.image);
    
    // 对于每个部分进行分类
    ch.convertTo(ch, CV_32FC1);
    int character = annClassifier->predict(ch);
    input.chars.push_back(Resources::cn_chars[character]);
    input.charsPos.push_back(cn_char.position);

    delete annClassifier;
}

bool OCR::ocr(Plate &input)
{
    if (DEBUG_MODE)
        std::cout << "Char regcognition..." << std::endl;

    //预处理
    preprocessPlate(input);
    // 字符分割
    std::vector<Char> segments = segment1(input);
    if (segments.empty())
    {
        return false;
    }
    
    // 训练中文分类器，并识别
    process_cn_char(input, segments[0]);
    // 训练字符分类器，并识别
    process_chars(input, segments);
}

} /* end for namespace pr */
