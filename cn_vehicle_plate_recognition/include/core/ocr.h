/* \file ocr.h
*  Implementation of Plate Recognition
*/

#pragma once

#include <string.h>
#include <vector>

#include <opencv2/ml/ml.hpp>

#include "../stdafx.h"

/* \namespace pr
*  Namespace where all the C++ Plate Recognition functionality resides
*/
namespace pr
{

class ANNClassifier;
class Plate;
class Char;
    
/* \class OCR
*  Implementation of Plate Recognition
*/
class OCR{

public:
    OCR();
    ~OCR();

public:
    bool ocr(Plate &input);

public:
    void preprocessPlate(Plate &plate);

private:    	
    cv::Mat removeMD(cv::Mat img);

    void process_chars(Plate &input, const std::vector<Char> &segments);
    void process_cn_char(Plate &input, const Char &cn_char);
};

} /* end for namespace pr */
