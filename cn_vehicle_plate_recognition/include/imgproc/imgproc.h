#pragma once

#include "stdafx.h"

/* \namespace pr
 * Namespace where all the C++ Plate Recognition functionality resides
*/
namespace pr
{

/*图像灰度化*/
void gray(cv::Mat& mat1, cv::Mat& mat2);

/*图像二值化*/
void threshold(cv::Mat& mat1, int threshold);

/*Soble边沿检测*/
void Sobel(cv::Mat& mat1, double x_beta);

/*直方图计算*/
int hist(cv::Mat& mat1, int number);

/*筛选步骤1*/
int** selection_Function_1(cv::Mat& mat1, int* number);

/*返回第一步裁减分割后图片*/
void pic_cutting(cv::Mat& mat1, cv::Mat& pic_cutting, int** selection, int number);

/*裁减出一张图片*/
void pic_cutting_1(cv::Mat& mat1, cv::Mat& mat2, cv::Point s1, cv::Point s2);

/*通过HSV颜色空间来检测该图片是否是车牌所在图片*/
int choice_Color(cv::Mat& mat1, int color_Start, int color_End);

/*在检测出车牌的图片中，框选车牌*/
int** car_License_box(cv::Mat& mat1, cv::Mat& mat2, int* number);

/*在框选出的方框中筛选出可能是车牌的方框*/
int box_selection(cv::Mat& mat1);
}
