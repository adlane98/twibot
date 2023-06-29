#ifndef TWIBOTCPP_UTILS_H
#define TWIBOTCPP_UTILS_H

#include <string>
#include <opencv2/core.hpp>

std::string getImageType(const cv::Mat &img, bool more_info = true);
torch::Tensor cvToTensor(cv::Mat &image);

#endif //TWIBOTCPP_UTILS_H
