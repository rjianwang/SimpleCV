#include <vector>

#include "../include/stdafx.h"
#include "../include/core/plate_detect.h"
#include "../include/core/plate.h"
#include "../include/core/ocr.h"
#include "../include/ml/svm.h"

using namespace pr;

void helper(int argc, char* argv[])
{
    std::cout << "========================================================================" << std::endl;;
    std::cout << std::endl;
    std::cout << "                   Chinese Vehicle Plate Recognition" << std::endl;
    std::cout << std::endl;
    std::cout << "========================================================================" << std::endl;;
    std::cout << std::endl;
    std::cout << "Usage: " << argv[0] << " [options] <image> " << std::endl; 
    std::cout << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "-debug:   Debug option. Provide this parameter for DEBUG mode." << std::endl;
    std::cout << "-detect:  Detect option. Provide this parameter for DETECT mode." << std::endl;
    std::cout << "========================================================================" << std::endl;
    std::cout << std::endl;
}

int main(int argc, char* argv[])
{
    ///////////////////////////////////////////////////////////////////////
    // 参数处理
    ///////////////////////////////////////////////////////////////////////
    helper(argc, argv);

    cv::Mat img;
    if (argc >= 1)
    {
        for (int i = 1; i < argc; i++)
        img = cv::imread(argv[argc - 1]);
        if (img.empty())
        {
            std::cout << "ERROR: image does not exist." << std::endl;
            return -1;
        }
    }
    else
    {
        std::cout << "ERROR: no input image." << std::endl;
        return -1;
    }

    std::vector<std::string> params;
    if (argc > 2)
    {
        for (int i = 1; i < argc - 1; i++)
            params.push_back(argv[i]);
    }

    if (!params.empty())
    {
        if (std::find(params.begin(), params.end(), "-debug") != params.end())
            DEBUG_MODE = true;
        if (std::find(params.begin(), params.end(), "-detect") != params.end())
            DETECT_MODE = true;
    }

    ////////////////////////////////////////////////////////////////////////
    // main process
    ////////////////////////////////////////////////////////////////////////

    // Detect and segment plates
    PlateDetection detector;
    detector.saveRecognition = false;
    std::vector<Plate> plates_temp = detector.segment(img);

    // SVM classifier
    SVMClassifier svmClassifier;
    //svmClassifier.load_model("SVM.xml");
    svmClassifier.load_data("../data/");
    svmClassifier.train();
    svmClassifier.save("SVM_cn.xml");

    // classify plates using SVM 
    std::vector<Plate> plates;
    for (int i = 0; i < plates_temp.size(); i++)
    {
        cv::Mat img = plates_temp[i].image;
        resize(img, img, cv::Size(36, 136), 0, 0, cv::INTER_CUBIC);
        img.convertTo(img, CV_32FC1);
        img = img.reshape(1, 1);
 //       int response = (int)svmClassifier.predict(img);

        time_t t;
        std::stringstream ss;
 //       ss << "./tmpPlate/" << response << "/" << time(&t) << i << ".jpg";
  //      cv::imwrite(ss.str(), plates_temp[i].image);

   //     if (response != 1)
   //         continue;

        plates.push_back(plates_temp[i]);
    }

    std::cout << "Num plates detected: " << plates.size() << "\n";
    std::string filename = "Plate";
    for (int i = 0; i < plates.size(); i++)
        cv::imshow(filename + (char)i + ".jpg", plates[i].image);
    
    // OCR
    if (!DETECT_MODE)
    {
        OCR ocr;
        for (int i = 0; i < plates.size(); i++)
        {
            Plate plate = plates[i];

            ocr.ocr(plate);
            std::string licensePlate = plate.str();
            std::cout << "License plate number: " << licensePlate << "\n";
            std::cout << "=============================================\n";
            cv::rectangle(img, plate.position, cv::Scalar(0, 0, 255), 2);
            cv::putText(img, licensePlate, cv::Point(plate.position.x, plate.position.y), CV_FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 200), 2);
            if (false)
                cv::imshow("Plate Detected seg", plate.image);
        }
    }

/*    for (int i = 0; i < plates.size(); i++)
        cv::rectangle(img, plates[i].position, cv::Scalar(0, 0, 255), 2);

    cv::imshow("Numbers of the Plate", img);*/

    while (cv::waitKey(0))
        break;
    cv::destroyAllWindows();

    return 0;
}
