#pragma once

#include <vector>
#include "../stdafx.h"


/* \namespace pr
*
*/
namespace pr
{
/*返回二值化图像mat1，满足至少发生number次跳变的开始行数*/
int detectionChange(cv::Mat& mat1, cv::Mat& mat2, int number);

/*将图片mat1 归一化到mat2( width,height大小)*/
void carCard_Resize(cv::Mat& mat1, cv::Mat& mat2, int width, int height);

/*垂直投影计算*/
void calcProj(cv::Mat& mat1, int* vArr, int number);

/*根据投影分割图片*/
int** projCut(int* vArr, int width, int* number);

/*检测传入图像是否是1*/
float pixelPercentage(cv::Mat& mat1);

} // end namespace pr
