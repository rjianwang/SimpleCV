/* \file plate_detect.cpp
 * Definition of class PlateDetection
 */

#include <time.h>

#include "../../include/core/plate_detect.h"
#include "../../include/core/plate.h"
#include "../../include/imgproc/imgproc.h"

/* \namespace pr
 * Namespace where all C++ Plate Recognition functionality resides
 */
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

    bool PlateDetection::verifySizes(const cv::RotatedRect &ROI)
    {
        // 中文车牌宽高比: 440 / 140 = 3.1429
        float aspect = 3.1429;
        // 设定区域面积的最小/最大尺寸，不在此范围内的不被视为车牌
        int min = 15 * 15 * aspect;    // min area
        int max = 144 * 144 * aspect;  // max area
        float rmin = 0.5 * aspect;
        float rmax = 3.5 * aspect;

        int area = ROI.size.height * ROI.size.width;
        float r = (float)ROI.size.width / (float)ROI.size.height;
        if (r < 1)
            r = (float)ROI.size.height / (float)ROI.size.width;


        // 判断是否符合以上参数
        bool result = true;
        if ((area < min || area > max) || (r < rmin || r > rmax))
            result = false;

        if (DEBUG_MODE)
            std::cout << "\tveryfi Size(" << result << "):"
                << "\tratio " << r
                << "\trmin " << rmin
                << "\trmax " << rmax
                << "\tarea " << area
                << "\tamax " << max
                << "\tamin " << min
                << std::endl;

        return result;
    }

    // 输入二值图像，检测图像中1的个数是否达到指定值
    bool PlateDetection::verifyNonZero(const cv::Mat &img, float thresh /* = 0.1 */)
    {
        int oneCount = cv::countNonZero(img);

        float ratio = oneCount / float(img.rows * img.cols);
        bool result = (ratio > thresh);

        if (DEBUG_MODE)
            std::cout << "\tverify None Zero Ratio(" << result 
                << "): " << ratio << std::endl;

        return result;

        return false;
    }

    // 输入灰度化车牌图像
    bool PlateDetection::verifyColorJump(const cv::Mat &img, int thresh /* = 7 */)
    {
        bool result = false;

        cv::Mat threshold;
        cv::threshold(img, threshold, 0, 255, CV_THRESH_OTSU + CV_THRESH_BINARY);

        int lineCount = 0;
        for (int i = 0; i < threshold.rows; i++)
        {
            int jumpCount = 0;
            for (int j = 0; j < threshold.cols - 1; j++)
            {
                if (threshold.at<char>(i, j) != threshold.at<char>(i, j + 1)) 
                    jumpCount += 1;
            }

            if (jumpCount >= thresh)
                lineCount++;
        }

        if (lineCount > 15)
            result = true;

        if (DEBUG_MODE)
        {
            std::cout << "\tverify Color Jump(" << result 
                << "): lines with matched color jump " << lineCount
                << std::endl;
            cv::imshow("candidate Plate", threshold);
        }

        return result;
    }

    cv::Mat PlateDetection::colorMatch(const cv::Mat &img, const char color,
            const bool adaptive_minsv /* = false */) 
    {
        // S和V的最小值由adaptive_minsv这个bool值判断
        // 如果为true，则最小值取决于H值，按比例衰减
        // 如果为false，则不再自适应，使用固定的最小值minabs_sv
        // 默认为false
        const float max_sv = 255;
        const float minref_sv = 64;
        const float minabs_sv = 95;

        // blue的H范围
        const int min_blue = 100;  // 100
        const int max_blue = 140;  // 140
        // yellow的H范围
        const int min_yellow = 15;  // 15
        const int max_yellow = 40;  // 40
        // white的H范围
        const int min_white = 0;   // 15
        const int max_white = 30;  // 40

        // 转到HSV空间进行处理
        // 颜色搜索主要使用的是H分量进行蓝色与黄色的匹配工作
        cv::Mat hsv;
        cv::cvtColor(img, hsv, CV_BGR2HSV);
        std::vector<cv::Mat> hsvSplit;
        cv::split(hsv, hsvSplit);
        cv::equalizeHist(hsvSplit[2], hsvSplit[2]);
        cv::merge(hsvSplit, hsv);

        //匹配模板基色,切换以查找想要的基色
        int min_h = 0;
        int max_h = 0;
        switch (color) {
            case 'b':
                min_h = min_blue;
                max_h = max_blue;
                break;
            case 'y':
                min_h = min_yellow;
                max_h = max_yellow;
                break;
            case 'w':
                min_h = min_white;
                max_h = max_white;
                break;
            default:
                // Color::UNKNOWN
                break;
        }

        float diff_h = float((max_h - min_h) / 2);
        float avg_h = min_h + diff_h;

        int channels = hsv.channels();
        int nRows = hsv.rows;

        //图像数据列需要考虑通道数的影响；
        int nCols = hsv.cols * channels;
        //连续存储的数据，按一行处理
        if (hsv.isContinuous()) 
        {
            nCols *= nRows;
            nRows = 1;
        }

        int i, j;
        uchar* p;
        float s_all = 0;
        float v_all = 0;
        float count = 0;
        for (i = 0; i < nRows; ++i) 
        {
            p = hsv.ptr<uchar>(i);
            for (j = 0; j < nCols; j += 3) 
            {
                int H = int(p[j]);      // 0-180
                int S = int(p[j + 1]);  // 0-255
                int V = int(p[j + 2]);  // 0-255

                s_all += S;
                v_all += V;
                count++;

                bool colorMatched = false;
                if (H > min_h && H < max_h) 
                {
                    float Hdiff = 0;
                    if (H > avg_h)
                        Hdiff = H - avg_h;
                    else
                        Hdiff = avg_h - H;

                    float Hdiff_p = float(Hdiff) / diff_h;

                    // S和V的最小值由adaptive_minsv这个bool值判断
                    // 如果为true，则最小值取决于H值，按比例衰减
                    // 如果为false，则不再自适应，使用固定的最小值minabs_sv
                    float min_sv = 0;
                    if (true == adaptive_minsv)
                        min_sv =  minref_sv - minref_sv / 2 *
                            (1 - Hdiff_p);  // inref_sv - minref_sv / 2 * (1 - Hdiff_p)
                    else
                        min_sv = minabs_sv;  // add

                    if ((S > min_sv && S < max_sv) && (V > min_sv && V < max_sv))
                        colorMatched = true;
                }

                if (colorMatched == true) 
                {
                    p[j] = 0;
                    p[j + 1] = 0;
                    p[j + 2] = 255;
                } 
                else 
                {
                    p[j] = 0;
                    p[j + 1] = 0;
                    p[j + 2] = 0;
                }
            }
        }

        // 获取颜色匹配后的二值灰度图
        cv::Mat gray;
        std::vector<cv::Mat> hsvSplit_done;
        cv::split(hsv, hsvSplit_done);
        gray = hsvSplit_done[2];

        return gray;
    }

    std::vector<Plate> PlateDetection::colorDetect(cv::Mat &img)
    {
        if (DEBUG_MODE)
            std::cout << "Color detect... " << std::endl;

        std::vector<Plate> plates;

        static int k = 0;
        std::stringstream ss;
        if (DEBUG_MODE)
        {
            cv::imshow("Sub-Image", img);
        }

        cv::Mat threshold;
        threshold = colorMatch(img, 'b');
        if (DEBUG_MODE)
            cv::imshow("THRESHOLD", threshold);

        plates = detectCore(img, threshold);

        if (DEBUG_MODE)
            std::cout << "Color detect result: " << plates.size() << std::endl;

        return plates;


        //for (int i = 0; i < 1; i++)
        {

            //cv::blur(threshold, threshold, cv::Size(5, 5));
            //cv::threshold(threshold, threshold, 0, 255, CV_THRESH_OTSU + CV_THRESH_BINARY);
            // 使用morphologyEx函数得到包含车牌的区域（但不包含车牌号）
            // 定义一个结构元素structuringElement，维度为17 * 3
            cv::Mat structuringElement = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(17, 3));
            cv::morphologyEx(threshold, threshold, CV_MOP_CLOSE, structuringElement);

            if (DEBUG_MODE)
            {
                cv::imshow(ss.str()  +  "-CLOSE", threshold);
            }
            /*
               cv::Mat structuringElement2 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(17, 3));
               cv::morphologyEx(threshold, threshold, CV_MOP_OPEN, structuringElement2);

               if (DEBUG_MODE)
               cv::imshow(ss.str()  + "-OPEN", threshold);
               */
            // 找到可能的车牌的轮廓
            std::vector<std::vector<cv::Point> > contours;
            findContours(threshold,
                    contours, // 检测的轮廓数组，每一个轮廓用一个point类型的vector表示
                    CV_RETR_EXTERNAL, // 表示只检测外轮廓
                    CV_CHAIN_APPROX_NONE); // 轮廓的近似办法，这里存储所有的轮廓点

            // 在白色的图上画出蓝色的轮廓
            cv::Mat result;
            img.copyTo(result);
            cv::drawContours(result,
                    contours,
                    -1,				    // 所有的轮廓都画出
                    cv::Scalar(255, 0, 0), // 颜色
                    3);		// 线粗
            if (DEBUG_MODE)
                cv::imshow(ss.str() + "-CONTOURS", result);

            // 对每个轮廓检测和提取最小区域的有界矩形区域
            std::vector<cv::RotatedRect> rects;
            std::vector<std::vector<cv::Point> >::iterator itc = contours.begin();
            while (itc != contours.end())
            {
                cv::RotatedRect ROI = cv::minAreaRect(cv::Mat(*itc));
                // 若没有达到设定的宽高比要求，移去该区域
                if (!verifySizes(ROI))
                    itc = contours.erase(itc);
                else
                {
                    ++itc;
                    rects.push_back(ROI);
                }
            }

            for (int k = 0; k < rects.size(); k++)
            {
                cv::Point2f rect_points[4]; 
                rects[k].points(rect_points);
                for (int j = 0; j < 4; j++)
                    line(result, rect_points[j], rect_points[(j + 1) % 4], cv::Scalar(0, 0, 255), 1, 8);

                // 得到旋转图像区域的矩阵
                float r = (float)rects[k].size.width / (float)rects[k].size.height;
                float angle = rects[k].angle;
                if (r < 1)
                    angle = 90 + angle;
                cv::Mat rotmat = cv::getRotationMatrix2D(rects[k].center, angle, 1);

                // 通过仿射变换旋转输入的图像
                cv::Mat img_rotated;
                cv::warpAffine(img, img_rotated, rotmat, img.size(), CV_INTER_CUBIC);

                // 最后裁剪图像
                cv::Size rect_size = rects[k].size;
                if (r < 1)
                    std::swap(rect_size.width, rect_size.height);
                cv::Mat img_crop;
                cv::getRectSubPix(img_rotated, rect_size, rects[k].center, img_crop);

                cv::Mat resultResized;
                resultResized.create(36, 136, CV_8UC3);
                resize(img_crop, resultResized, resultResized.size(), 0, 0, cv::INTER_CUBIC);

                // 为了消除光照影响，对裁剪图像使用直方图均衡化处理
                cv::Mat grayResult;
                cv::cvtColor(resultResized, grayResult, CV_BGR2GRAY);
                cv::blur(grayResult, grayResult, cv::Size(3, 3));
                grayResult = histeq(grayResult);

                if (verifyColorJump(grayResult, 15)) //颜色跳变检测
                    //if (verifyNonZero(grayResult, 0.3))
                {
                    if (saveRecognition){
                        std::stringstream ss(std::stringstream::in | std::stringstream::out);
                        ss << "tmp/" << filename << "_" << k << ".jpg";
                        imwrite(ss.str(), grayResult);
                    }
                    plates.push_back(Plate(grayResult, rects[k].boundingRect()));

                    if (DEBUG_MODE)
                        cv::imshow("Plate - Color", grayResult);
                }
            }
        }
        if (DEBUG_MODE)
            std::cout << "\tColor detect result: " << plates.size() << std::endl;
        //return plates;
    }

    cv::Mat PlateDetection::preprocessImg(cv::Mat &img)
    {
        // 图像转换为灰度图
        cv::Mat gray;
        // cv::cvtColor(img, gray, CV_BGR2GRAY);
        bgr2gray(img, gray);

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
            cv::imshow("SOBEL", sobel);

        // 阈值分割得到二值图像，所采用的阈值由Otsu算法得到
        cv::Mat threshold;
        cv::threshold(sobel, threshold, 0, 255, CV_THRESH_OTSU + CV_THRESH_BINARY);

        return threshold;
    }

    std::vector< std::vector<int> > PlateDetection::colorJump(cv::Mat &img)
    {
        // 图像预处理，得到二值化的图像
        cv::Mat mat;
        cv::cvtColor(img, mat, CV_BGR2GRAY);
        cv::threshold(mat, mat, 0, 255, CV_THRESH_OTSU + CV_THRESH_BINARY);

        bool flag = false;

        std::vector< std::vector<int> > ret;
        for (int i = 0; i < mat.rows; i++)
        {
            int jumpCount = 0;
            for (int j = 0; j < mat.cols - 1; j++)
            {
                if (mat.at<char>(i, j) != mat.at<char>(i, j + 1)) 
                    jumpCount += 1;
            }

            int a;
            if (jumpCount >= 7 && flag == false)
            {
                a = i;
                flag = true;
            }
            else if (jumpCount < 7 && flag == true)
            {
                if (i - a < 8)  continue;

                ret.push_back({a, i - 1});
                flag = false;
            }
        }
        return ret;
    }

    std::vector<Plate> PlateDetection::sobelDetect(cv::Mat &img)
    {
        std::vector<Plate> plates;

        if (DEBUG_MODE)
            std::cout << "Sobel detect..." << std::endl;

        static int num_img = 0;
        if (DEBUG_MODE)
        {
            cv::imshow("Sub-Image", img);
        }

        cv::Mat threshold = preprocessImg(img);
        if (DEBUG_MODE)
            cv::imshow("THRESHOLD", threshold);

        plates = detectCore(img, threshold);

        if (DEBUG_MODE)
            std::cout << "Sobel detect result: " << plates.size() << std::endl;
        return plates;
    }

    // 返回值为灰度图
    cv::Mat PlateDetection::segment(const cv::Mat &img, const cv::RotatedRect &rect)
    {
        cv::Mat result;
        img.copyTo(result);

        std::cout << rect.angle << ", " << rect.center << ", " << rect.size << std::endl;
                cv::Point2f rect_points[4]; 
                rect.points(rect_points);
                for (int j = 0; j < 4; j++)
                    line(result, rect_points[j], rect_points[(j + 1) % 4], cv::Scalar(0, 0, 255), 1, 8);

                // 得到旋转图像区域的矩阵
                float r = (float)rect.size.width / (float)rect.size.height;
                float angle = rect.angle;
                if (r < 1)
                    angle = 90 + angle;
                cv::Mat rotmat = cv::getRotationMatrix2D(rect.center, angle, 1);

                // 通过仿射变换旋转输入的图像
                cv::Mat img_rotated;
                cv::warpAffine(result, img_rotated, rotmat, result.size(), CV_INTER_CUBIC);

                // 最后裁剪图像
                cv::Size rect_size = rect.size;
                if (r < 1)
                    std::swap(rect_size.width, rect_size.height);
                cv::Mat img_crop;
                cv::getRectSubPix(img_rotated, rect_size, rect.center, img_crop);
                std::cout << img_crop.size() << std::endl;

                cv::Mat resultResized;
                resultResized.create(36, 136, CV_8UC3);
                resize(img_crop, resultResized, resultResized.size(), 0, 0, cv::INTER_CUBIC);

                // 为了消除光照影响，对裁剪图像使用直方图均衡化处理
                cv::Mat grayResult;
                cv::cvtColor(resultResized, grayResult, CV_BGR2GRAY);
                cv::blur(grayResult, grayResult, cv::Size(3, 3));
                grayResult = histeq(grayResult);

        return grayResult;
    }

    // 输入为二值图像
    std::vector<Plate> PlateDetection::detectCore(const cv::Mat &img, const cv::Mat &threshold)
    {
        std::vector<Plate> plates;

        // 使用morphologyEx函数得到包含车牌的区域（但不包含车牌号）
        // 定义一个结构元素structuringElement，维度为17 * 3
        cv::Mat structuringElement = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(17, 3));
        cv::morphologyEx(threshold, threshold, CV_MOP_CLOSE, structuringElement);

        if (DEBUG_MODE)
            cv::imshow("CLOSE", threshold);

        for (int i = 0; i < 10; i++)
        {
            if (i > 0 && i <= 5)
            {
                if (!verifyNonZero(threshold, 0.09))
                    break;
                cv::Mat structuringElement2 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(7, 7));
                cv::morphologyEx(threshold, threshold, CV_MOP_OPEN, structuringElement2);
            }
            else if (i >= 6)
            {
                if (!verifyNonZero(threshold, 0.09))
                    break;
                cv::morphologyEx(threshold, threshold, CV_MOP_CLOSE, structuringElement);
            }

            // 找到可能的车牌的轮廓
            std::vector<std::vector<cv::Point> > contours;
            std::vector<cv::RotatedRect> rects;
            findContours(threshold,
                    contours, // 检测的轮廓数组，每一个轮廓用一个point类型的vector表示
                    CV_RETR_EXTERNAL, // 表示只检测外轮廓
                    CV_CHAIN_APPROX_NONE); // 轮廓的近似办法，这里存储所有的轮廓点

            std::vector<std::vector<cv::Point> >::iterator itc = contours.begin();
            float max_ratio = 0.0;
            while (itc != contours.end())
            {
                // 对每个轮廓检测和提取最小区域的有界矩形区域
                cv::RotatedRect ROI = cv::minAreaRect(cv::Mat(*itc));

                float ratio = ROI.size.width / (float)ROI.size.height;
                if (ratio < 1)
                    ratio = ROI.size.height / (float)ROI.size.width;
                if (ratio > max_ratio)
                    max_ratio = ratio;

                // 若没有达到设定的宽高比要求，移去该区域
                if (!verifySizes(ROI))
                    itc = contours.erase(itc);
                else
                {
                    ++itc;
                    rects.push_back(ROI);
                }
            }

            if (max_ratio > 6) // 矩形长宽比大于6则进行膨胀操作
            {
                cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
                cv::dilate(threshold, threshold, element);
                continue;
            }

            // 在白色的图上画出蓝色的轮廓
            cv::Mat result;
            img.copyTo(result);
            cv::drawContours(result,
                    contours,
                    -1,				    // 所有的轮廓都画出
                    cv::Scalar(255, 0, 0), // 颜色
                    3);		// 线粗
            if (DEBUG_MODE)
                cv::imshow("CONTOURS", result);

            for (int k = 0; k < rects.size(); k++)
            {
                cv::Mat grayResult = segment(img, rects[k]);
                if (verifyColorJump(grayResult, 15))
                {
                    if (saveRecognition)
                    {
                        std::stringstream ss(std::stringstream::in | std::stringstream::out);
                        ss << "tmp/" << filename << "_" << k << ".jpg";
                        imwrite(ss.str(), grayResult);
                    }

                    if (DEBUG_MODE)
                    {
                        std::string filename = "Plate";
                        cv::imshow(filename + char(k), grayResult);
                    }
                    plates.push_back(Plate(grayResult, rects[k].boundingRect()));
                }
            } 
            if (plates.size() > 0)
                break;
        }
        return plates;
    }

    std::vector<Plate> PlateDetection::detect(cv::Mat &img)
    {
        std::vector<Plate> plates;

        // 进行图像颜色跳变检测
        std::vector< std::vector<int> > color_jump = colorJump(img);

        for (int i = 0; i < color_jump.size(); i++)
        {
            if (color_jump[i][1] - color_jump[i][0] < 15)
                continue;   // 如果高度小于一定值，则抛弃

            cv::Rect rect(0, 
                    color_jump[i][0],
                    img.cols,
                    color_jump[i][1] - color_jump[i][0]);
            cv::Mat mat(img, rect); // 图像裁剪

            std::vector<Plate> temp1 = sobelDetect(mat); // 轮廓检测方法
            if (DEBUG_MODE && cv::waitKey(0))
            {
                cv::destroyAllWindows();
            }
            std::vector<Plate> temp2 = colorDetect(mat); // 颜色检测方法
            if (DEBUG_MODE && cv::waitKey(0))
            {
                cv::destroyAllWindows();
            }

            if (!temp2.empty())
            {
                for (int k = 0; k < temp2.size(); k++)
                {
                    temp2[k].position.y += color_jump[i][0];
                }
                plates.insert(plates.end(), temp2.begin(), temp2.end());
            }
            else if (!temp1.empty())
            {
                for (int k = 0; k < temp1.size(); k++)
                    temp1[k].position.y = color_jump[i][0];
                plates.insert(plates.end(), temp1.begin(), temp1.end());
            }

        }
        return plates;
    }

} /* end for namespace pr */
