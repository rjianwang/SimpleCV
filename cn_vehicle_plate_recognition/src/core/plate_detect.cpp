/* \file plate_detect.cpp
 * Implementation of class PlateDetection
 */

#include <time.h>

#include "../../include/core/plate_detect.h"
#include "../../include/core/plate.h"

namespace pr
{

/* \class PlateDetection
 * Implementation of plate detect methods
 */
PlateDetection::PlateDetection()
{
    saveRecognition = false;
}


PlateDetection::~PlateDetection()
{
}

cv::Mat PlateDetection::histeq(cv::Mat img)
{
    cv::Mat imt(img.size(), img.type());
    // 若输入图像为彩色，需要在HSV空间中做直方图均衡处理
    // 再转换回RGB格式
    if (img.channels() == 3)
    {
        cv::Mat hsv;
        std::vector<cv::Mat> hsvSplit;
        cv::cvtColor(img, hsv, CV_BGR2HSV);
        cv::split(hsv, hsvSplit);
        cv::equalizeHist(hsvSplit[2], hsvSplit[2]);
        cv::merge(hsvSplit, hsv);
        cv::cvtColor(hsv, imt, CV_HSV2BGR);
    }
    // 若输入图像为灰度图，直接做直方图均衡处理
    else if (img.channels() == 1){
        equalizeHist(img, imt);
    }

    return imt;
}

bool PlateDetection::verifySizes(cv::Rect ROI)
{
    // 以下设置车牌默认参数，用于识别矩形区域内是否为目标车牌
    float error = 0.4;
    // 中文车牌宽高比: 440 / 140 = 3.1429
    float aspect = 3.1429;
    // 设定区域面积的最小/最大尺寸，不在此范围内的不被视为车牌
    int min = 15 * 15 * aspect;    // min area
    int max = 144 * 144 * aspect;  // max area
    float rmin = aspect - aspect * error;
    float rmax = aspect + aspect * error;

    int area = ROI.height * ROI.width;
    float r = (float)ROI.width / (float)ROI.height;
    if (r < 1)
        r = (float)ROI.height / (float)ROI.width;

    // 判断是否符合以上参数
    if ((area < min || area > max) || (r < rmin || r > rmax))
        return false;

    return true;
}

bool PlateDetection::verifySizes(cv::RotatedRect ROI)
{
    // 以下设置车牌默认参数，用于识别矩形区域内是否为目标车牌
    float error = 0.4;
    // 中文车牌宽高比: 440 / 140 = 3.1429
    float aspect = 3.1429;
    // 设定区域面积的最小/最大尺寸，不在此范围内的不被视为车牌
    int min = 15 * 15 * aspect;    // min area
    int max = 144 * 144 * aspect;  // max area
    float rmin = aspect - aspect * error;
    float rmax = aspect + aspect * error;

    int area = ROI.size.height * ROI.size.width;
    float r = (float)ROI.size.width / (float)ROI.size.height;
    if (r < 1)
        r = (float)ROI.size.height / (float)ROI.size.width;

    // 判断是否符合以上参数
    if ((area < min || area > max) || (r < rmin || r > rmax))
        return false;

    return true;
}

std::vector<Plate> PlateDetection::segment(cv::Mat img)
{
    std::vector<Plate> plates;

    // 图像转换为灰度图
    cv::Mat gray;
    cv::cvtColor(img, gray, CV_BGR2GRAY);

    // 均值滤波，去噪
    cv::blur(gray, gray, cv::Size(5, 5));

    // Sobel算子检测边缘
    cv::Mat sobel;
    Sobel(gray,			    // 输入图像
            sobel,          // 输出图像
            CV_8U,			// 输出图像的深度
            1,				// x方向上的差分阶数
            0,				// y方向上的差分阶数
            3,				// 扩展Sobel核的大小，必须是1,3,5或7
            1,				// 计算导数值时可选的缩放因子，默认值是1
            0,				// 表示在结果存入目标图之前可选的delta值，默认值为0
            cv::BORDER_DEFAULT); // 边界模式，默认值为BORDER_DEFAULT

    if (DEBUG_MODE)
        cv::imshow("Sobel", sobel);

    // 阈值分割得到二值图像，所采用的阈值由Otsu算法得到
    cv::Mat threshold;
    cv::threshold(sobel, threshold, 0, 255, CV_THRESH_OTSU + CV_THRESH_BINARY);

    if (DEBUG_MODE)
        cv::imshow("Threshold Image", threshold);

    // 使用morphologyEx函数得到包含车牌的区域（但不包含车牌号）
    // 定义一个结构元素structuringElement，维度为17 * 3
    cv::Mat structuringElement = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(17, 3));
    cv::morphologyEx(threshold, threshold, CV_MOP_CLOSE, structuringElement);

    if (DEBUG_MODE)
        cv::imshow("Close", threshold);

/*    cv::Mat structuringElement2 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(17, 3));
    for (int i = 0; i < 100; i++)
    {
        cv::blur(threshold, threshold, cv::Size(7, 7));
        cv::morphologyEx(threshold, threshold, CV_MOP_OPEN, structuringElement2);
        cv::threshold(threshold, threshold, 0, 255, CV_THRESH_OTSU + CV_THRESH_BINARY);
    }*/

    if (DEBUG_MODE)
        cv::imshow("Open", threshold);

    // 找到可能的车牌的轮廓
    std::vector<std::vector<cv::Point> > contours;
    findContours(threshold,
            contours, // 检测的轮廓数组，每一个轮廓用一个point类型的vector表示
            CV_RETR_EXTERNAL, // 表示只检测外轮廓
            CV_CHAIN_APPROX_NONE); // 轮廓的近似办法，这里存储所有的轮廓点

    // 对每个轮廓检测和提取最小区域的有界矩形区域
    std::vector<std::vector<cv::Point> >::iterator itc = contours.begin();
    std::vector<cv::RotatedRect> rects;
    // 若没有达到设定的宽高比要求，移去该区域
    while (itc != contours.end())
    {
        cv::RotatedRect ROI = cv::minAreaRect(cv::Mat(*itc));
        if (!verifySizes(ROI)){
            itc = contours.erase(itc);
        }
        else{
            ++itc;
            rects.push_back(ROI);
        }
    }

    // 在白色的图上画出蓝色的轮廓
    cv::Mat result;
    img.copyTo(result);
    cv::drawContours(result,
            contours,
            -1,				    // 所有的轮廓都画出
            cv::Scalar(255, 0, 0), // 颜色
            2);		// 线粗

    for (int i = 0; i < rects.size(); i++)
    {
        if (!verifySizes(rects[i].boundingRect()))
            continue;

        cv::Point2f rect_points[4]; 
        rects[i].points(rect_points);
        for (int j = 0; j < 4; j++)
            line(result, rect_points[j], rect_points[(j + 1) % 4], cv::Scalar(0, 0, 255), 1, 8);

        // 得到旋转图像区域的矩阵
        float r = (float)rects[i].size.width / (float)rects[i].size.height;
        float angle = rects[i].angle;
        if (r < 1)
            angle = 90 + angle;
        cv::Mat rotmat = cv::getRotationMatrix2D(rects[i].center, angle, 1);

        // 通过仿射变换旋转输入的图像
        cv::Mat img_rotated;
        cv::warpAffine(img, img_rotated, rotmat, img.size(), CV_INTER_CUBIC);

        // 最后裁剪图像
        cv::Size rect_size = rects[i].size;
        if (r < 1)
            std::swap(rect_size.width, rect_size.height);
        cv::Mat img_crop;
        cv::getRectSubPix(img_rotated, rect_size, rects[i].center, img_crop);

        cv::Mat resultResized;
        resultResized.create(36, 136, CV_8UC3);
        resize(img_crop, resultResized, resultResized.size(), 0, 0, cv::INTER_CUBIC);

        // 为了消除光照影响，对裁剪图像使用直方图均衡化处理
        cv::Mat grayResult;
        cv::cvtColor(resultResized, grayResult, CV_BGR2GRAY);
        cv::blur(grayResult, grayResult, cv::Size(3, 3));
        grayResult = histeq(grayResult);
        if (saveRecognition){
            std::stringstream ss(std::stringstream::in | std::stringstream::out);
            ss << "tmp/" << filename << "_" << i << ".jpg";
            imwrite(ss.str(), grayResult);
        }
        plates.push_back(Plate(grayResult, rects[i].boundingRect()));
    }
/*
    // 使用漫水填充算法裁剪车牌获取更清晰的轮廓
    for (int i = 0; i< rects.size(); i++)
    {
        cv::circle(result, rects[i].center, 3, cv::Scalar(0, 255, 0), -1);
        // 得到宽度和高度中较小的值，得到车牌的最小尺寸
        float minSize = (rects[i].size.width < rects[i].size.height) ? rects[i].size.width : rects[i].size.height;
        minSize = minSize - minSize * 0.1;
        // 在块中心附近产生若干个随机种子
        srand(time(NULL));
        // 初始化漫水填充算法的参数
        cv::Mat mask;
        mask.create(img.rows + 2, img.cols + 2, CV_8UC1);
        mask = cv::Scalar::all(0);
        // loDiff表示当前观察像素值与其部件邻域像素值或者待加入
        // 该部件的种子像素之间的亮度或颜色之负差的最大值
        int loDiff = 30;
        // upDiff表示当前观察像素值与其部件邻域像素值或者待加入
        // 该部件的种子像素之间的亮度或颜色之正差的最大值
        int upDiff = 30;
        int connectivity = 4; // 用于控制算法的连通性，可取4或者8
        int newMaskVal = 255;
        int NumSeeds = 10;
        cv::Rect ccomp;
        // 操作标志符分为几个部分
        int flags = connectivity + // 用于控制算法的连通性，可取4或者8
            (newMaskVal << 8) +
            CV_FLOODFILL_FIXED_RANGE + // 设置该标识符，会考虑当前像素与种子像素之间的差
            CV_FLOODFILL_MASK_ONLY; // 函数不会去填充改变原始图像, 而是去填充掩模图像
        for (int j = 0; j < NumSeeds; j++){
            cv::Point seed;
            seed.x = rects[i].center.x + rand() % (int)minSize - (minSize / 2);
            seed.y = rects[i].center.y + rand() % (int)minSize - (minSize / 2);
            circle(result, seed, 1, cv::Scalar(0, 255, 255), -1);
            // 运用填充算法，参数已设置
            int area = floodFill(img,
                    mask,
                    seed,
                    cv::Scalar(255, 0, 0),
                    &ccomp,
                    cv::Scalar(loDiff, loDiff, loDiff),
                    cv::Scalar(upDiff, upDiff, upDiff),
                    flags);
        }

        if (DEBUG_MODE)
            cv::imshow("MASK", mask);

        // 得到裁剪掩码后，检查其有效尺寸
        // 对于每个掩码的白色像素，先得到其位置
        // 再使用minAreaRect函数获取最接近的裁剪区域
        std::vector<cv::Point> pointsInterest;
        cv::Mat_<uchar>::iterator itMask = mask.begin<uchar>();
        cv::Mat_<uchar>::iterator end = mask.end<uchar>();
        for (; itMask != end; ++itMask)
            if (*itMask == 255)
                pointsInterest.push_back(itMask.pos());

        cv::RotatedRect minRect = cv::minAreaRect(pointsInterest);

        if (verifySizes(minRect)){
            // 旋转矩形图
            cv::Point2f rect_points[4]; minRect.points(rect_points);
            for (int j = 0; j < 4; j++)
                line(result, rect_points[j], rect_points[(j + 1) % 4], cv::Scalar(0, 0, 255), 1, 8);

            // 得到旋转图像区域的矩阵
            float r = (float)minRect.size.width / (float)minRect.size.height;
            float angle = minRect.angle;
            if (r < 1)
                angle = 90 + angle;
            cv::Mat rotmat = cv::getRotationMatrix2D(minRect.center, angle, 1);

            // 通过仿射变换旋转输入的图像
            cv::Mat img_rotated;
            cv::warpAffine(img, img_rotated, rotmat, img.size(), CV_INTER_CUBIC);

            // 最后裁剪图像
            cv::Size rect_size = minRect.size;
            if (r < 1)
                std::swap(rect_size.width, rect_size.height);
            cv::Mat img_crop;
            cv::getRectSubPix(img_rotated, rect_size, minRect.center, img_crop);

            cv::Mat resultResized;
            resultResized.create(33, 144, CV_8UC3);
            resize(img_crop, resultResized, resultResized.size(), 0, 0, cv::INTER_CUBIC);

            // 为了消除光照影响，对裁剪图像使用直方图均衡化处理
            cv::Mat grayResult;
            cv::cvtColor(resultResized, grayResult, CV_BGR2GRAY);
            cv::blur(grayResult, grayResult, cv::Size(3, 3));
            grayResult = histeq(grayResult);
            if (saveRecognition){
                std::stringstream ss(std::stringstream::in | std::stringstream::out);
                ss << "tmp/" << filename << "_" << i << ".jpg";
                imwrite(ss.str(), grayResult);
            }
            plates.push_back(Plate(grayResult, minRect.boundingRect()));
        }
    }*/
    if (DEBUG_MODE)
        cv::imshow("Contours", result);

    return plates;
}

} /* end for namespace pr */
