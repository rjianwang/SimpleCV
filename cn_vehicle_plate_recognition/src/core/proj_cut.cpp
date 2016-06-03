#include <stdlib.h>
#include "../../include/core/proj_cut.h"

namespace pr
{

int detectionChange(cv::Mat& mat1, cv::Mat& mat2, int number)
{
	IplImage pI_1 = mat1, pI_2;
	cv::Scalar s1, s2;
	int width = mat1.rows;
	int height = mat1.cols;
	int sum = 0, sum_2 = 0, width_1 = 0, width_2 = 0;
	int i, j;

	for(i = 0; i < width; i++){
		sum = 0;
		sum_2 = 0;
		for(j = 0; j < height-1; j++)
		{
			s1 = cvGet2D(&pI_1, i, j);
			s2 = cvGet2D(&pI_1, i, j+1);
			if(((int)s1.val[0]) != ((int)s2.val[0]))
			{
				sum += 1;
				sum_2 = 0;
			}
			else
			{
				sum_2 += 1;	
			}
			if(sum_2 != 0)
			{
				if(height / sum_2 < 5)
				{
					sum = 0;
					break;
				}
			}
		}
		
		if(sum >= number)
		{
			width_1 = i;
			break;
		}
		else
		{
			width_1 = i;	
		}
	}

	for(i = width - 1; i > 0; i--)
	{
		sum = 0;
		sum_2 = 0;
		for(j = 0; j < height - 1; j++)
		{
			s1 = cvGet2D(&pI_1, i, j);
			s2 = cvGet2D(&pI_1, i, j + 1);
			if(((int)s1.val[0]) != ((int)s2.val[0]))
			{
				sum += 1;
				sum_2 = 0;
			}
			else
			{
				sum_2 += 1;	
			}
			if(sum_2 != 0)
			{
				if(height / sum_2 < 1)
				{
					sum = 0;
					break;
				}
			}
		}
		if(sum >= number)
		{
			width_2 = i;
			break;	
		}
		else
		{
			width_2 = i;
		}
	}
	if(width_2 <= width_1)
	{
		width_2 = width;	
	}
	mat2 = cv::Mat(width_2 - width_1 + 1, height, CV_8UC1, 1);
	pI_2 = mat2;
	for(i = width_1; i <= width_2; i++)
	{
		for(j = 0; j < height; j++)
		{
			s1 = cvGet2D(&pI_1, i, j);
			cvSet2D(&pI_2, i - width_1, j, s1);
		}	
	}
}



float pixelPercentage(cv::Mat& mat1)
{
	IplImage pI_1 = mat1;
	cv::Scalar s1;
	int width = mat1.rows;
	int height = mat1.cols;
	int i, j;
	float sum = 0, allSum = 0, tmp;

	for(i=0; i < width; i++)
	{
		for(j = 0; j < height; j++)
		{
			s1 = cvGet2D(&pI_1, i, j);
			if(s1.val[0] > 20)
			{
				sum += 1;
			}
			allSum += 1;
		}	
	}
	tmp = sum / allSum;

	return tmp;
}
} // end namespace pr
