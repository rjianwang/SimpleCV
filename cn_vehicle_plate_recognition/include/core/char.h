/* \file char.h
 *  This file includes Char Segmentation Algorithoms.
 */

#pragma once

#include <vector>
#include "../stdafx.h"

class Plate;

/* \namespace pr
 *	Namespace where all the C++ Plate Recognition functionality resides
 */
namespace pr
{

    /* \class Char
     *  A Char object is a result of char segmentation.
     */
    class Char
    {
        public:
            Char();
            Char(cv::Mat img, cv::Rect pos);
            cv::Mat image;
            cv::Rect position;
    };

    // 字符大小
    const cv::Size charSize(12, 24);

    // 根据字符大小比例等进行预判断
    bool verifySizes(cv::Mat r);
    // 字符预处理
    cv::Mat preprocessChar(cv::Mat in);

    /* \algorithms
     *  基于轮廓检测的字符分割
     */
    // 字符分割 
    std::vector<Char> segment1(const Plate &input);
    // 获取特殊字符（车牌中的第二个字符） 
    int getSpecificChar(const Plate &plate, const std::vector<Char> &input);
    // 根据特殊字符提取中文字符（车牌中的第一个字符） 
    Char getChineseChar(const cv::Mat &img, const Char &spec);
    // 合并检测到的轮廓，删除过小的轮廓
    void mergeContours(std::vector<Char> &segments);


    /* \algorithms
     *  基于投影的字符分割
     */
    // 字符分割
    std::vector<Char> segment2(const Plate &input);
    // 竖直投影计算 
    cv::Mat vhist(const cv::Mat &img);
    // 根据投影分割图片 
    std::vector< std::vector<int> > hsegment(const cv::Mat &hist);
    // 水平投影计算
    cv::Mat hhist(const cv::Mat &img);
    // 根据水平投影分割图片
    std::vector<int> vsegment(const cv::Mat &hist);
    // 合并水平投影，区域间隔小于1
    void mergeHist(std::vector<std::vector<int>> &segments);

} /* end for namespace pr */
