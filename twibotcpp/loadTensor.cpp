#include "loadTensor.h"

#include <torch/torch.h>

using Module = torch::jit::script::Module;

std::vector<char> getBytes(const std::string& fileName) {
    std::ifstream input(fileName, std::ios::binary);
    std::vector<char> bytes(
            (std::istreambuf_iterator<char>(input)),
            (std::istreambuf_iterator<char>()));

    input.close();
    return bytes;
}

torch::Tensor loadTensor(const std::string& fileName)
{
    std::vector<char> f = getBytes(fileName);
    torch::IValue x = torch::pickle_load(f);
    at::Tensor my_tensor = x.toTensor();
    return my_tensor;
}

void warmup(Module model)
{
    at::Tensor warmup_tensor = at::ones({ 1, 3, 640, 640}, at::kCUDA);
    model.forward({ warmup_tensor });
}

