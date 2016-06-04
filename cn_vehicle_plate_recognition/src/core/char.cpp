/* \file char.cpp
 *  This file includes Char Segmentation Algorithoms.
 */

#include "../../include/core/plate.h"
#include "../../include/core/char.h"
#include "../../include/tool/tool.h"

/* \namespace pr
 * Namespace where all C++ Plate Recognition functionality reside
 */
namespace pr
{

    /* \Class char
     *  A Char object is a result of char segmentation
     */
    Char::Char()
    {

    }

    Char::Char(cv::Mat img, cv::Rect pos)
    {
        this->image = img;
        this->position = pos;
    }

    /* 根据字符大小比例等进行预判断 */
    bool verifySizes(cv::Mat r)
    {
        bool result = false; //判断结果

        // 字符宽高比为45/77
        float aspect = 45.0f / 77.0f;
        float charAspect = (float)r.cols / (float)r.rows;
        float minHeight = 15;
        float maxHeight = 35;
        // 最大宽高比及最小宽高比
        float minAspect = 0.1;
        float maxAspect = 1;
        // 非零值个数
        float area = countNonZero(r);
        // 字符区域的大小
        float bbArea = r.cols * r.rows;
        // 非零值所占的比例
        float percPixels = area / bbArea;

        if (percPixels < 0.9 && 
                charAspect > minAspect && 
                charAspect < maxAspect && 
                r.rows >= minHeight && 
                r.rows < maxHeight)
            result = true;
        
        if (DEBUG_MODE)
            std::cout << "\tverify Size(" << result << "): "
                << " Char aspect " << charAspect 
                << " [" << minAspect << ", " << maxAspect << ", " << aspect << "] " 
                << " None zero ratio " << percPixels 
                << " Char height " << r.rows 
                << std::endl;

        return result;
    }

    // 字符预处理
    cv::Mat preprocessChar(cv::Mat in)
    {
        int h = in.rows;
        int w = in.cols;
        cv::Mat transformMat = cv::Mat::eye(2, 3, CV_32F);
        int m = std::max(w, h);
        transformMat.at<float>(0, 2) = m / 2 - w / 2;
        transformMat.at<float>(1, 2) = m / 2 - h / 2;

        cv::Mat warpImage(m, m, in.type());
        cv::warpAffine(in, warpImage, transformMat, warpImage.size(), cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0));

        cv::Mat out;
        cv::resize(warpImage, out, cv::Size(charSize, charSize));

        return out;
    }


    /* \algorithms
     *  基于轮廓的字符分割
     */

    // 字符分割 
    // input 为经过二值化预处理的plate 
    std::vector<Char> segment1(const Plate &input)
    {
        if (DEBUG_MODE)
        {
            std::cout << "Segmenting..." << std::endl;
        }
        std::vector<Char> segments(7);

        cv::Mat img_contours;
        input.image.copyTo(img_contours);

        // 找到可能的车牌轮廓
        std::vector< std::vector<cv::Point> > contours;
        cv::findContours(img_contours,
                contours, // 检测的轮廓数组，每一个轮廓用一个point类型的vector表示
                CV_RETR_EXTERNAL, // 只检测外轮廓
                CV_CHAIN_APPROX_NONE); //轮廓的近似办法，这里存储所有的轮廓点

        // 在白色的图上画出轮廓
        cv::Mat result1;
        input.image.copyTo(result1);
        cv::cvtColor(result1, result1, CV_GRAY2RGB);
        cv::drawContours(result1, contours,
                -1,  // 所有的轮廓都画出
                cv::Scalar(255, 0, 0), // 颜色
                1); // 线粗

        cv::Mat result2;
        result1.copyTo(result2);

        // 对每个轮廓检测和提取最小区域的有界矩形区域
        std::vector<std::vector<cv::Point> >::iterator itc = contours.begin();
        std::vector<Char> output;
        int i = 0;
        while (itc != contours.end()) 
        {
            cv::Rect mr = cv::boundingRect(cv::Mat(*itc));
            cv::rectangle(result1, mr, cv::Scalar(0, 125, 255));
            // 裁剪图像
            cv::Mat auxRoi(input.image, mr);
            // 若没有达到设定的宽高比，则移去该区域
            if (verifySizes(auxRoi)){
                auxRoi = preprocessChar(auxRoi);
                output.push_back(Char(auxRoi, mr));
            }
            ++itc;
        }

        // 按x坐标排序
        qsort(output, 0, output.size() - 1);
        
        // 获得特殊字符
        int specIndex = getSpecificChar(input, output);
        if (specIndex != 1)
            return output;    // 如果为-1，则返回不继续，需对该返回值进行判断

        // 根据特殊字符分割除中文字符（车牌中的第一个字符）外的所有字符
        for (int i = specIndex, j = 1; i < output.size() && j <= 6; i++, j++)
            segments[j] = output[i];

        // 根据特殊字符分割中文字
        segments[0] = getChineseChar(input.image, output[specIndex]);

        char res[20];
        for (int i = 0; i < segments.size(); i++)
        {
            // 保存分割结果
            sprintf(res, "PlateNumber%d.jpg", i);
            cv::imwrite(res, segments[i].image);
            // 画出字符分割的轮廓
            cv::rectangle(result2, segments[i].position, cv::Scalar(0, 125, 255));
        }
        if (DEBUG_MODE)
        {
            std::cout << "Spec index: " << specIndex << std::endl;
            std::cout << "Num chars: " << segments.size() << std::endl;
            cv::imshow("Segmented Chars by contour", result1);
            cv::imshow("Segmented Chars by spec char", result2);
        }

        if (cv::waitKey(0))
            cv::destroyAllWindows();

        return segments;
    }

    /* 获取特殊字符（车牌中的第二个字符） */
    int getSpecificChar(const Plate &plate, const std::vector<Char> &input)
    {
        if (DEBUG_MODE)
            std::cout << "Find specific char..." << std::endl;

        int maxHeight = 0, maxWidth = 0;

        for (int i = 0; i < input.size(); i++)
        {
            if (input[i].position.height > maxHeight)
                maxHeight = input[i].position.height;
            if (input[i].position.width > maxWidth)
                maxWidth = input[i].position.width;
        }

        int i;
        for (i = 0; i < input.size(); i++)
        {
            cv::Rect mr = input[i].position;
            int midx = mr.x + mr.width / 2;
            
            int high = int(plate.width / 7) * 2;
            int low = int(plate.width / 7);

            if (DEBUG_MODE)
                std::cout << "\tmidx: " << midx
                    << " low: " << low
                    << " high: " << high
                    << " width: " << mr.width
                    << " maxWidth * 80%: " << maxWidth * 0.8
                    << " height: " << mr.height
                    << " maxHeight * 80%: " << maxHeight * 0.8
                    << std::endl;

            if ((mr.width >= maxWidth * 0.8 || mr.height >= maxHeight * 0.8)
                    && (midx <= high)
                    && (midx >= low))             
                return i;   // specific char
        }

        return -1;
    }

    /* 根据特殊字符提取中文字符（车牌中的第一个字符） */
    Char getChineseChar(const cv::Mat &img, const Char &spec)
    {
        Char cn_char;

        cn_char.position.height = spec.position.height;
        cn_char.position.width = spec.position.width * 1.15f;
        cn_char.position.y = spec.position.y;

        int x = spec.position.x - int(cn_char.position.width * 1.15f);
        cn_char.position.x = x > 0 ? x : 0;

        cv::Mat auxRoi(img, cn_char.position);
        cn_char.image = auxRoi;

        return cn_char;
    }


    /* \algorithms
     *  基于投影的字符分割
     */
    /* 字符分割*/
    std::vector<Char> segment2(Plate input)
    {

    }

    /* 计算垂直投影 */
    cv::Mat calcProj(const cv::Mat &img)
    {
        cv::Mat vhist = cv::Mat::zeros(1, img.cols, CV_32F);

        for(int j = 0; j < img.cols; j++)
        {
            cv::Mat data = img.col(j);
            vhist.at<float>(0, j) = cv::countNonZero(data);
        }

        return vhist;
    }

    /* 根据投影分割图片 */
    std::vector< std::vector<int> > projSegment(cv::Mat &vhist)
    {
        std::vector< std::vector<int> > ret;

        int flag = 0;
        int threshold = 2;

        for(int i = 0; i < vhist.cols; i++)
        {
            std::vector<int> value(2);
            if(	(vhist.at<float>(0, i) <= threshold)      &&
                    (vhist.at<float>(0, i + 1) > threshold))
            {
                value[0] = i;
                flag = 1;
            }
            else if ( (vhist.at<float>(0, i) > threshold)    &&
                    (vhist.at<float>(0, i + 1) <= threshold) &&
                    flag)
            {
                value[1] = i;
                flag = 0;

                ret.push_back(value);
            }		
        }

        return ret;
    }

} /* end for namespace pr */
