#include <iostream>

#include <opencv2/imgproc.hpp>

#include <torch/torch.h>

#include "utils.h"

std::string getImageType(const cv::Mat& img, bool more_info)
{
    std::string r;
    int type = img.type();
    uchar depth = type & CV_MAT_DEPTH_MASK;
    uchar chans = 1 + (type >> CV_CN_SHIFT);

    switch (depth) {
        case CV_8U:  r = "8U"; break;
        case CV_8S:  r = "8S"; break;
        case CV_16U: r = "16U"; break;
        case CV_16S: r = "16S"; break;
        case CV_32S: r = "32S"; break;
        case CV_32F: r = "32F"; break;
        case CV_64F: r = "64F"; break;
        default:     r = "User"; break;
    }

    r += "C";
    r += chans + '0';

    if (more_info)
        std::cout << "depth: " << img.depth() << " channels: " << img.channels() << std::endl;

    return r;
}

torch::Tensor cvToTensor(cv::Mat& image)
{
    cv::Mat preTensorImage;
    cv::resize(image, preTensorImage, cv::Size(360, 640));
    torch::Tensor imageTensor = torch::from_blob(preTensorImage.data, { preTensorImage.rows, preTensorImage.cols, 3 });
    imageTensor = imageTensor.transpose(2, 0);
    imageTensor = imageTensor.unsqueeze(0);
    imageTensor = torch::constant_pad_nd(imageTensor, torch::IntList{ 0, 0, 140, 140, 0, 0 }, 114);
    imageTensor = imageTensor.to(torch::kFloat);
    imageTensor = imageTensor.to(torch::kCUDA);
    imageTensor = imageTensor.div(255);
    std::cout << imageTensor.sizes() << std::endl;
    torch::Tensor indexedTensor = imageTensor.index({ 0, 1, "..." });

    return imageTensor;
}