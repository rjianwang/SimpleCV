/* \file Plate.cpp
 * Implementation of Class Plate
 */

#include "../../include/core/plate.h"

/* \namespace pr
 * Namespace where all C++ Plate Recognition functionality resides
 */
namespace pr
{

/* \class Plate
 * 
 */
Plate::Plate(){
}

Plate::Plate(cv::Mat img, cv::Rect pos){
	image = img;
	position = pos;

    width = 136;
    height = 36;
}

std::string Plate::str()
{
    std::string result = "";
    std::vector<int> orderIndex;
    std::vector<int> xpositions;
	for (int i = 0; i < charsPos.size(); i++)
    {
		orderIndex.push_back(i);
		xpositions.push_back(charsPos[i].x);
	}
	float min = xpositions[0];
	int minIdx = 0;
	for (int i = 0; i < xpositions.size(); i++)
    {
		min = xpositions[i];
		minIdx = i;
		for (int j = i; j < xpositions.size(); j++)
        {
			if (xpositions[j] < min)
            {
				min = xpositions[j];
				minIdx = j;
			}
		}
		int aux_i = orderIndex[i];
		int aux_min = orderIndex[minIdx];
		orderIndex[i] = aux_min;
		orderIndex[minIdx] = aux_i;

		float aux_xi = xpositions[i];
		float aux_xmin = xpositions[minIdx];
		xpositions[i] = aux_xmin;
		xpositions[minIdx] = aux_xi;
	}
	for (int i = 0; i < orderIndex.size(); i++)
    {
		result = result + chars[orderIndex[i]];
	}
	return result;
}

} /* end for namespace pr */
