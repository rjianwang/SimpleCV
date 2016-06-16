/* \file ocr.cpp
 *  Implementation of Plate Recognition
 */

#include <string>
#include "../../include/core/plate.h"
#include "../../include/core/resource.h"
#include "../../include/core/char.h"
#include "../../include/core/ocr.h"
#include "../../include/core/ocr2.h"
#include "../../include/core/feature.h"
#include "../../include/tool/tool.h"
#include "../../include/ml/svm.h"
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
        // 去除水平方向边缘或铆钉
        removeFringe(threshold);

        if (DEBUG_MODE)
            cv::imshow("Threshold plate", threshold);

        plate.image = threshold;
    }

    // 输入为二值化的车牌图像
    // 去除水平方向的白色边框或铆钉
    cv::Mat OCR::removeFringe(cv::Mat img)
    {
        if (DEBUG_MODE)
        {
            std::cout << "Remove mao ding..." << std::endl;
        }

        int line = 4;
        for (int i = 0; i < line; i++)
        {
            int color_jump = 0;
            for (int j = 0; j < img.cols - 1; j++)
                if (img.at<char>(i, j) != img.at<char>(i, j))
                    color_jump++;

            if (color_jump < 5)
                img.row(i).setTo(cv::Scalar(0));
        }

        for (int i = img.rows - line; i < img.rows; i++)
        {
            int color_jump = 0;
            for (int j = 0; j < img.cols; j++)
                if (img.at<char>(i, j) != img.at<char>(i, j))
                    color_jump++;

            if (color_jump < 5)
                img.row(i).setTo(cv::Scalar(0));
        }
        return img;
    }

    void OCR::process_chars(Plate &input, const std::vector<Char> &segments)
    {
        // ANN Classifier for digits and letters
        SVMClassifier *svmClassifier = new SVMClassifier();

        svmClassifier->load_char("../data/charSamples/");
        svmClassifier->train();

        for (int i = 1; i < segments.size(); i++){
            // 对每个字符进行预处理，使得对所有图像均有相同的大小 
            cv::Mat ch = preprocessChar(segments[i].image);
            cv::imshow("Char", ch);
            if (cv::waitKey(0))
                cv::destroyAllWindows();

            // 对于每个部分进行分类
            ch.convertTo(ch, CV_32FC1);
            ch = ch.reshape(1, 1);
            //cv::Mat f = features(ch);
            int character = svmClassifier->predict(ch);
            std::cout << character << std::endl;
            input.chars.push_back(std::string(1, Resources::chars[character]));
            input.charsPos.push_back(segments[i].position);
        }

        delete svmClassifier;
    }

    void OCR::process_cn(Plate &input, const Char &cn_char)
    {
        // ANN Classifier for Chinese Characters
        SVMClassifier *svmClassifier = new SVMClassifier();
        svmClassifier->load_cn("../data/cn_chars/"); 
        svmClassifier->train();

        // 对字符进行预处理
        cv::Mat ch = preprocessChar(cn_char.image);

        // 对于每个部分进行分类
        ch.convertTo(ch, CV_32FC1);
        cv::Mat f = features(ch);
        int character = svmClassifier->predict(f);
        input.chars.push_back(Resources::cn_chars[character]);
        input.charsPos.push_back(cn_char.position);

        delete svmClassifier;
    }

    bool OCR::ocr(Plate &input)
    {
        if (DEBUG_MODE)
            std::cout << "Char regcognition..." << std::endl;

        //预处理
        preprocessPlate(input);
        // 字符分割
        std::vector<Char> segments = segment2(input);

        if (segments.size() != 7)
            segments = segment1(input);
        if (segments.size() != 7)
            return false;

        // 训练中文分类器，并识别
//        process_cn(input, segments[0]);
        // 训练字符分类器，并识别
//        process_chars(input, segments);

        std::vector<int> result1 = ocr1();
        std::vector<int> result2 = ocr2();

        
        input.chars.push_back((Resources::cn_chars[result1[0]]));
        input.charsPos.push_back(segments[0].position);

        for (int i = 0; i < result2.size(); i++)
        {
            std::stringstream ss;
            ss << Resources::chars[result2[i]];
            input.chars.push_back(ss.str());
            input.charsPos.push_back(segments[i].position);
        }
        return true;
    }

} /* end for namespace pr */
