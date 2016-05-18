#include "StdAfx.h"

#include <vector>
#include "ImageSegmentation.h"
#include "Plate.h"
#include "SVMClassifier.h"
#include "OCR.cpp"

int main(int argc, char* argv[])
{
    cv::Mat img = cv::imread("plates/2715DTZ.jpg");

    // Detect and segment plate
    ImageSegmentation detector;
    detector.saveRecognition = false;
    detector.DEBUG = false;
    std::vector<Plate> plates_temp = detector.segment(img);

    // Load training data
    cv::FileStorage fs("SVM.xml", cv::FileStorage::READ);
    cv::Mat trainData;
    cv::Mat labelData;
    fs["TrainingData"] >> trainData;
    fs["classes"] >> labelData;

    // SVM classifier
    SVMClassifier svmClassifier;
    svmClassifier.train(trainData, labelData);

    // classify plates using SVM 
    std::vector<Plate> plates;
    for (int i = 0; i < plates_temp.size(); i++)
    {
        cv::Mat img = plates_temp[i].image;
        cv::Mat p = img.reshape(1, 1);
        p.convertTo(p, CV_32FC1);
        int response = (int)svmClassifier.predict(p);
        if (response == 1)
            plates.push_back(plates_temp[i]);
    }

    std::cout << "=============================================\n";
    std::cout << "Num plates detected: " << plates.size() << "\n";

    // OCR
    OCR ocr;
    ocr.saveSegments = true;
    ocr.DEBUG = false;
    for (int i = 0; i < plates.size(); i++)
    {
        Plate plate = plates[i];

        std::string plateNumber = ocr.ocr(&plate);
        std::string licensePlate = plate.str();
        std::cout << "=============================================\n";
        std::cout << "License plate number: " << licensePlate << "\n";
        std::cout << "=============================================\n";
        cv::rectangle(img, plate.position, cv::Scalar(0, 0, 200));
        cv::putText(img, licensePlate, cv::Point(plate.position.x, plate.position.y), CV_FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 200), 2);
        if (false)
            cv::imshow("Plate Detected seg", plate.image);
    }

    cv::imshow("Numbers of the Plate", img);

    while (cv::waitKey(0))
        break;
    cv::destroyAllWindows();

    fs.release();
    
    return 0;
}
