/* \file ann.h
 * An implementation of ANN Classifier
 */

#pragma once

#include "../stdafx.h"
#include <string>
#include <opencv2/ml/ml.hpp>

#define HORIZONTAL  1
#define VERTICAL    0

/* \namespace pr
 * Namespace where all C++ Plate Recognition functionality resides
 */
namespace pr
{

    /* \class ANNClassifier
     * An customized ANN classifier
     */
    class ANNClassifier
    {
        public:
            ANNClassifier(int num_neurons, int num_output);
            ~ANNClassifier();

        public:
            void train();
            int predict(const cv::Mat &sample);

            void load(const std::string filename);
            void load_cn(const std::string filepath);
            void load_data(const std::string filepath);

        private:
            CvANN_MLP ann;
            CvANN_MLP_TrainParams params;
            cv::Mat trainData;
            cv::Mat labelData;

        private:
            int num_neurons;
            int num_output;

    }; /* ends for class ANNClassifier */ 

} /* ends for namespace pr */
