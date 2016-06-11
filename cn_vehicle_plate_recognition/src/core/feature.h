/* \file feature.cpp
 * Compute features of characters
 */

#include "../include/core/feature.h"

/* \namespace pr
 * Namespace where all C++ Plate Recognition functionality reside
 */
namesapce pr
{

    cv::Mat features(const cv::Mat &img)
    {
        
    }

    std::vector<float> gradientF(const cv::Mat &img)
    {
        std::vector<float> features;

        cv::Mat gray;
        cv::cvtColor(img, gray, CV_BGR2GRAY);

        float mask[3][3] = {{1, 2, 1},
                    {0, 0, 0},
                    {-1, -2, -1}};
        cv::Mat y_mask = cv::Mat(3, 3, CV_32F, mask);
        cv::Mat x_mask = y_mask.t();
        cv::Mat sobelX, sobelY;

        cv::filter2D(img, sobelX, CV_32F, x_mask);
        cv::filter2D(img, sobelY, CV_32F, y_mask);

        sobelX = cv::abs(sobelX);
        sobelY = cv::abs(sobelY);

        float totalX = sumMat(sobelX);
        float totalY = sumMat(sobelY);

        for (int i = 0; i < img.rows; i = i + 4;)
        {
            for (int j = 0; j < img.cols; j = j + 4)
            {
                cv::Mat subX = sobelX(cv::Rect(j, i, 4, 4)); 
                featuers.push_back(sumMat(subX) / totalX);
                cv::Mat subY = sobelY(cv::Rect(j, i, 4, 4));
                features.push_back(sumMat(subX) / totalY);
            }
        }

        return features;
    }
 
    float sumMat(const cv::Mat &img)
    {
        float sumValue = 0.0;
        int rows = img.rows;
        int cols = img.cols;
        if (image.isContinuours())
        {
            cols = rows * cols;
            rows = 1;
        }

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                sumValue += img.at<uchar>(i, h);
            }
        }
    }

} /* end for namespace pr *
