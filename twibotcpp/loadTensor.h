#ifndef TWIBOTCPP_LOADTENSOR_H
#define TWIBOTCPP_LOADTENSOR_H

#include <string>

#include <torch/script.h>
#include <torch/torch.h>

torch::Tensor loadTensor(const std::string& fileName);
void warmup(torch::jit::script::Module model);


#endif //TWIBOTCPP_LOADTENSOR_H
