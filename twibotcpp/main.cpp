#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <torch/torch.h>
#include <torch/script.h>

#include "utils.h"
#include "loadTensor.h"

using namespace cv;
using namespace std;
using Module = torch::jit::script::Module;

void postProcess(const torch::Tensor& detections, float confThres, float iouThres)
{
    constexpr int numClasses = 2;
    int batchSize = detections.size(0);
    torch::Tensor confMask = detections.select(2, 4).ge(confThres).unsqueeze(2);

    for (int batchIndex = 0; batchIndex < batchSize; batchIndex++)
    {
        torch::Tensor det = torch::masked_select(detections[batchIndex], confMask).view({-1, 7});
        cout << "Det size after mask : " << det.sizes() << endl;
    }

}

torch::Tensor inference(const torch::Tensor& tensorImage, Module model)
{
    model.eval();
    vector<torch::jit::IValue> inputs;
    inputs.emplace_back(tensorImage.to(torch::kCUDA));

    torch::jit::IValue output = model.forward({ tensorImage });

    auto detections = output.toTuple()->elements()[0].toTensor();

    cout << "Taille de la detection : " << detections.sizes() << endl;

    postProcess(detections, 0.25, 0.5);

    return torch::rand(1);
}

void processImage(const Module& model)
{
    string image_path = "/home/adlane/projets/twinit-dataset/video-8/9.jpg";
    cv::Mat image = cv::imread(image_path);
    torch::Tensor imageTensor = cvToTensor(image);
    cout << imageTensor[0][0][150][458] << endl;
    torch::Tensor pred = inference(imageTensor, model);
}


int main(int argc, char** argv )
{
    cout << "Is CUDA available ? ";
    if (torch::cuda::is_available())
        cout << "Yes" << endl;
    else
    {
        cout << "No, end of the program" << endl;
        return -1;
    }

    string modulePath = "/home/adlane/projets/twibot/yolov5/runs/train/exp4/weights/best.torchscript";
    torch::Device device(torch::kCUDA);
    Module model = torch::jit::load(modulePath, device);
    warmup(model);

    processImage(model);

    return EXIT_SUCCESS;
}
