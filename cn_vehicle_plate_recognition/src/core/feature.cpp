/* \file feature.cpp
 * Compute features of characters
 */

#include "../../include/core/feature.h"

/* \namespace pr
 * Namespace where all C++ Plate Recognition functionality reside
 */
namespace pr
{

    cv::Mat features(const cv::Mat &img)
    {
        cv::Mat features(1, 48, CV_32FC1);
        std::vector<float>  f1 = gradientF(img);
        cv::Mat f2 = pixelF(img);

        int i = 0;
        for (; i < f1.size(); i++)
        {
            features.at<float>(0, i) = f1[i];
        }
        for (int j = 0; j < f2.cols; i++, j++)
        {
            features.at<float>(0, i) = f2.at<uchar>(0, j);
        }

        return features;        
    }

    // 输入为二值化字符图像
    std::vector<float> gradientF(const cv::Mat &img)
    {
        std::vector<float> features;

        cv::Mat resized;
        cv::resize(img, resized, cv::Size(8, 16));

        float mask[3][3] = {{1, 2, 1},
                    {0, 0, 0},
                    {-1, -2, -1}};
        cv::Mat y_mask = cv::Mat(3, 3, CV_32F, mask) / 8;
        cv::Mat x_mask = y_mask.t();
        cv::Mat sobelX, sobelY;

        cv::filter2D(resized, sobelX, CV_32F, x_mask);
        cv::filter2D(resized, sobelY, CV_32F, y_mask);

        sobelX = cv::abs(sobelX);
        sobelY = cv::abs(sobelY);

        for (int i = 0; i < resized.rows; i = i + 4)
        {
            for (int j = 0; j < resized.cols; j = j + 4)
            {
                cv::Mat subX = sobelX(cv::Rect(j, i, 4, 4)); 
                features.push_back(cv::countNonZero(subX));
                cv::Mat subY = sobelY(cv::Rect(j, i, 4, 4));
                features.push_back(cv::countNonZero(subX));
            }
        }

        return features;
    }

    // 输入为二值化字符图像
    cv::Mat pixelF(const cv::Mat &img)
    {
        cv::Mat resized;
        cv::resize(img, resized, cv::Size(4, 8));
        resized = resized.reshape(1, 1);

        return resized;
    }

} /* end for namespace pr */
