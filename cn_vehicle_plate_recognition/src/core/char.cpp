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

    // 根据字符大小比例等进行预判断
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
        cv::resize(warpImage, out, charSize);

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
            std::cout << "Segmenting..." << std::endl;
        
        std::vector<Char> segments(7);

        cv::Mat threshold;
        input.image.copyTo(threshold);

        // 找到可能的车牌轮廓
        std::vector< std::vector<cv::Point> > contours;
        cv::findContours(threshold,
                contours, // 检测的轮廓数组，每一个轮廓用一个point类型的vector表示
                CV_RETR_EXTERNAL, // 只检测外轮廓
                CV_CHAIN_APPROX_NONE); //轮廓的近似办法，这里存储所有的轮廓点

        // 在白色的图上画出轮廓
        cv::Mat result1;
        input.image.copyTo(result1);
        cv::drawContours(result1, contours,
                -1,  // 所有的轮廓都画出
                cv::Scalar(255, 0, 0), // 颜色
                1); // 线粗

        // 对每个轮廓检测和提取最小区域的有界矩形区域
        std::vector<std::vector<cv::Point> >::iterator itc = contours.begin();
        
        std::vector<Char> output;
        int i = 0;
        while (itc != contours.end()) 
        {
            cv::Rect mr = cv::boundingRect(cv::Mat(*itc));
            // 裁剪图像
            cv::Mat auxRoi(threshold, mr);
            if (verifySizes(auxRoi)){
                //auxRoi = preprocessChar(auxRoi);
                output.push_back(Char(auxRoi, mr));
                cv::rectangle(result1, mr, cv::Scalar(0, 255, 0));
                // 保存
                std::stringstream ss;
                ss << "PlateNumber" << i++ << ".jpg";
                cv::imwrite(ss.str().c_str(), auxRoi);
            }
            ++itc;
        }

        if (DEBUG_MODE)
            cv::imshow("Char Segment by Contours", result1);

        // 按x坐标排序
        qsort(output, 0, output.size() - 1);
        // 合并轮廓，同时删除过小的轮廓
        //mergeContours(output);

        // 获得特殊字符
        int specIndex = getSpecificChar(input, output);
        if (specIndex != 1)
        {
            if (DEBUG_MODE && cv::waitKey(0))
                cv::destroyAllWindows();
            return output;    // 如果为-1，则返回不继续，需对该返回值进行判断
        }

        // 根据特殊字符分割除中文字符（车牌中的第一个字符）外的所有字符
        for (int i = specIndex, j = 1; i < output.size() && j <= 6; i++, j++)
            segments[j] = output[i];

        // 根据特殊字符分割中文字
        segments[0] = getChineseChar(input.image, output[specIndex]);

        cv::Mat result2;
        result1.copyTo(result2);
        for (int i = 0; i < segments.size(); i++)
        {
            // 保存分割结果
            std::stringstream ss;
            ss << "PlateNumber" << i << ".jpg";
            cv::imwrite(ss.str().c_str(), segments[i].image);
            // 画出字符分割的轮廓
            cv::rectangle(result2, segments[i].position, cv::Scalar(0, 125, 255));
        }
        if (DEBUG_MODE)
        {
            std::cout << "Spec index: " << specIndex << std::endl;
            std::cout << "Num chars: " << segments.size() << std::endl;
            cv::imshow("Segmented Chars by spec char", result2);
        }

        if (cv::waitKey(0))
            cv::destroyAllWindows();

        return segments;
    }

    // 合并检测到的轮廓，删除过小的轮廓
    void mergeContours(std::vector<Char> &segments)
    {
        for (int i = 0; i < segments.size() - 1; i++)
        {
            if (segments[i + 1].position.x - (segments[i].position.width + segments[i].position.x) < 3)
            {
                if (segments[i].position.width < 8 || segments[i + 1].position.width < 8)
                {
                    segments[i].position.width = segments[i + 1].position.width + 
                        segments[i + 1].position.x - segments[i].position.x;
                    segments[i].position.y = std::min(segments[i].position.y, segments[i + 1].position.y);
                    segments[i].position.height = std::max(segments[i].position.height, segments[i + 1].position.height);
                }
            }

            // 若没有达到设定的宽高比，则移去该区域
            if (!verifySizes(segments[i].image)){
                segments.erase(segments.begin() + i);
            }
        }
    }

    // 获取特殊字符（车牌中的第二个字符） 
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
    // i字符分割
    // 输入为二值化的车牌图像
    std::vector<Char> segment2(const Plate &input)
    {
        if (DEBUG_MODE)
            std::cout << "Segment by hist..." << std::endl;
        std::vector<Char> segments;

        cv::Mat threshold = input.image;

        cv::Mat ver_hist = vhist(threshold);
        cv::Mat hor_hist = hhist(threshold);

        // 获得水平及垂直分区
        std::vector<std::vector<int>> hsegments = hsegment(ver_hist);
        std::vector<int> vsegments = vsegment(hor_hist);
        // 合并垂直分区
        mergeHist(hsegments);

        // 根据垂直分区，获得个字符的x坐标及宽度
        for (int i = 0; i < hsegments.size(); i++)
        {
            int x = hsegments[i][0];
            int width = hsegments[i][1] - x;
            int y = vsegments[0];
            int height = vsegments[1] - y;

            if (width < 13) 
                width = (x + 13 < threshold.cols) ? 13 : width;
            if (width >= 20)
                width = 15;
            cv::Rect rect(x, y, width, height);
            cv::Mat img(threshold, rect);

            segments.push_back({img, rect});
        }

        cv::Mat result;
        input.image.copyTo(result);
        for (int i = 0; i < segments.size(); i++)
        {
            // 保存分割结果
            std::stringstream ss;
            ss << "PlateNumber" << i << ".jpg";
            cv::imwrite(ss.str(), segments[i].image);
            cv::rectangle(result, segments[i].position, cv::Scalar(255));
        }
        if (DEBUG_MODE)
        {
            cv::imshow("Char Segmented by Hist", result);
            if (cv::waitKey(0))
                cv::destroyAllWindows();
            std::cout << "Num chars: " << segments.size() << std::endl;
        }

        return segments;
    }

    // 计算垂直投影
    cv::Mat vhist(const cv::Mat &img)
    {
        if (DEBUG_MODE)
            std::cout << "\tCalculate Verticle Hist..." << std::endl;
        cv::Mat hist = cv::Mat::zeros(1, img.cols, CV_32F);

        for(int j = 0; j < img.cols; j++)
        {
            cv::Mat data = img.col(j);
            hist.at<float>(0, j) = cv::countNonZero(data);
        }

        if (DEBUG_MODE)
            std::cout << "\t" << hist << std::endl;

        return hist;
    }

    cv::Mat hhist(const cv::Mat &img)
    {
        if (DEBUG_MODE)
            std::cout << "\tCalculate Horizontal Hist..." << std::endl;

        cv::Mat hist = cv::Mat::zeros(1, img.rows, CV_32F);
        for (int i = 0; i < img.rows; i++)
            hist.at<float>(0, i) = cv::countNonZero(img.row(i));

        if (DEBUG_MODE)
        {
            std::cout << "Hello, world!" << std::endl;
            std::cout << "\t" << hist << std::endl;
        }
        return hist;
    }

    // 根据投影分割图片，获得x坐标及width
    std::vector< std::vector<int> > hsegment(const cv::Mat &vhist)
    {
        if (DEBUG_MODE)
            std::cout << "\tHorizontal Segment..." << std::endl;

        std::vector< std::vector<int> > ret;

        int flag = 0;
        int threshold = 1;

        int i, a = 0;
        for(i = 0; i < vhist.cols - 1; i++)
        {
            if(	(flag == 0 && 
                    vhist.at<float>(0, i) <= threshold)      &&
                    vhist.at<float>(0, i + 1) > threshold)
            {
                a = i;
                flag = 1;
            }
            if (vhist.at<float>(0, i) > threshold    &&
                    vhist.at<float>(0, i + 1) <= threshold &&
                    flag == 1)
            {
                flag = 0;
                ret.push_back({a, i + 1});
            }		
        }

        if (flag == 1)
            ret.push_back({a, i});

        return ret;
    }

    // 根据水平投影获得y坐标和高度等信息
    std::vector<int> vsegment(const cv::Mat &hist)
    {
        if (DEBUG_MODE)
            std::cout << "\tVerticle Segment..." << std::endl;

        std::vector<int> ret = {0, 35};
        bool flag = false;
        int threshold = 5;
        int i;
        for (i = 0; i < hist.cols - 1; i++)
        {
            if (i > 20)
                flag = true;

            if (flag == false && 
                    hist.at<float>(0, i) <= threshold &&
                    hist.at<float>(0, i + 1) > threshold
                    )
            {
                flag = true;
                ret[0] = i;
            }

            if (flag == true && 
                    hist.at<float>(0, i) > threshold &&
                    hist.at<float>(0, i + 1) <= threshold
                    )
            {
                flag = false;
                ret[1] = i + 1;
            }
        }

        return ret;
    }

    // 合并垂直分区
    void mergeHist(std::vector<std::vector<int>> &segments)
    {
        if (DEBUG_MODE)
            std::cout << "\tMerge Verticle Hist..." << std::endl;

        int threshold = 4;
        std::vector<std::vector<int>> ret;
        for (int i = 0; i < segments.size() - 1; i++)
        {
            if (segments[i + 1][0] - segments[i][1] < 3)
            {
                if (segments[i + 1][1] - segments[i + 1][0] < 5)
                {
                    segments[i][1] = segments[i + 1][1];
                    segments.erase(segments.begin() + i + 1);
                }
            }

            if (segments[i][1] - segments[i][0] < 5)
                segments.erase(segments.begin() + i);
        }
    }

} /* end for namespace pr */
