// twibot-cpp2.cpp : Ce fichier contient la fonction 'main'. L'exécution du programme commence et se termine à cet endroit.
//

#include <iostream>
#include <iomanip>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/highgui.hpp>

#include <torch/torch.h>
#include <torch/script.h>
#include <ATen/ATen.h>
#include <ATen/Functions.h>

using namespace std;
using Module = torch::jit::script::Module;


std::vector<char> get_the_bytes(std::string filename) {
    std::ifstream input(filename, std::ios::binary);
    std::vector<char> bytes(
        (std::istreambuf_iterator<char>(input)),
        (std::istreambuf_iterator<char>()));

    input.close();
    return bytes;
}


at::Tensor load_tensor(std::string filename)
{
    std::vector<char> f = get_the_bytes(filename);
    torch::IValue x = torch::pickle_load(f);
    at::Tensor my_tensor = x.toTensor();
    return my_tensor;
}



void SaveCsv(torch::Tensor t, const std::string& filepath, const std::string& separator = ",")
{
    t = t.flatten(1).contiguous().cpu();
    float* ptr = t.data_ptr<float>();

    std::ofstream csvHandle(filepath);
    for (size_t i = 0; i < t.sizes()[0]; ++i)
    {
        for (size_t k = 0; k < t.sizes()[1]; ++k)
        {
            float number = (*ptr++);
            csvHandle << std::setprecision(7) << +number;

            if (k < (t.sizes()[1] - 1))
            {
                csvHandle << separator;
            }
        }

        csvHandle << "\n";
    }
}



std::string get_image_type(const cv::Mat& img, bool more_info = true)
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
    r += (chans + '0');

    if (more_info)
        std::cout << "depth: " << img.depth() << " channels: " << img.channels() << std::endl;

    return r;
}


torch::Tensor nonMaxSuppression(torch::Tensor prediction, float conf_thresh = 0.25, float iou_thresh = 0.25)
{
    int nc = 2;

}


void warmup(Module model)
{
    at::Tensor warmup_tensor = at::ones({ 1, 3, 384, 640 }, at::kCUDA);
    model.forward({ warmup_tensor });
}


at::Tensor cvToTensor(cv::Mat& image)
{
    cv::Mat preTensorImage;
    cv::resize(image, preTensorImage, cv::Size(360, 640));
    at::Tensor imageTensor = at::from_blob(preTensorImage.data, { preTensorImage.rows, preTensorImage.cols, 3 }, at::kByte);
    imageTensor = imageTensor.transpose(2, 0);
    imageTensor = imageTensor.unsqueeze(0);
    imageTensor = torch::constant_pad_nd(imageTensor, torch::IntList{ 0, 0, 12, 12, 0, 0 }, 114);
    imageTensor = imageTensor.to(torch::kFloat);
    imageTensor = imageTensor.to(torch::kCUDA);
    imageTensor = imageTensor.div(255);
    at::Tensor indexedTensor = imageTensor.index({ 0, 1, "..." });

     return imageTensor;
}

at::Tensor inference(at::Tensor tensorImage, Module model)
{
    model.eval();
    c10::IValue inf = model.forward({ tensorImage });
    auto x = (*inf.toTuple().get());
    /*cout << x.isList() << endl;
    cout << inf.isCustomClass() << endl;
    cout << inf.isDouble() << endl;
    cout << inf.isEnum() << endl;
    cout << inf.isTensor() << endl;
    cout << inf.isTuple() << endl;
    cout << inf.isObject() << endl;
    cout << inf.isNone() << endl;*/
    //auto t = x[0];
    auto a = x.elements();
    auto b = a[0];
    auto c = b.toTuple()->elements()[0];
    auto d = b.toTuple()->elements()[1];
    cout << c.isList() << endl;
    cout << c.isCustomClass() << endl;
    cout << c.isDouble() << endl;
    cout << c.isEnum() << endl;
    cout << c.isTensor() << endl;
    cout << c.isTuple() << endl;
    cout << c.isObject() << endl;
    cout << c.isNone() << endl << endl;

    cout << d.isList() << endl;
    cout << d.isCustomClass() << endl;
    cout << d.isDouble() << endl;
    cout << d.isEnum() << endl;
    cout << d.isTensor() << endl;
    cout << d.isTuple() << endl;
    cout << d.isObject() << endl;
    cout << d.isNone() << endl;
    auto z = *x.elements()[0].toTuple();
    cout << c.toTensor().sizes() << endl;
    cout << d.toTensor().sizes() << endl;

    at::Tensor tensor1 = c.toTensor();
    at::Tensor tensor2 = c.toTensor();

    cout << tensor1[0][67][1].item<float>() << endl;
    cout << tensor2[0][67][1].item<float>() << endl;


    int y = 1;
    return torch::rand(1);
}


int processVideo(Module model)
{
    cv::VideoCapture video("E:\\twibot\\twinit-dataset\\DSC_3686.MOV");

    if (!video.isOpened())
    {
        cout << "Error while opening video stream" << endl;
        return -1;
    }

    while (1)
    {
        cv::Mat frame;
        video >> frame;

        if (frame.empty()) break;

        at::Tensor tensorFrame = cvToTensor(frame);
        at::Tensor pred = inference(tensorFrame, model);

        cv::imshow("Frame", frame);

        if ((char)cv::waitKey(25) == 27)
            return 0;

    }
}


void processImage(Module model)
{
    string image_path = "E:\\twibot\\twinit-dataset\\video-8\\9.jpg";
    cv::Mat image = cv::imread(image_path);
    at::Tensor imgt = at::from_blob(image.data, { image.rows, image.cols, 3 }, at::kByte);
    at::Tensor imageTensor = cvToTensor(image);
    at::Tensor loadedTensor = load_tensor("E:\\twibot\\twinit-dataset\\debug\\py_tensor.pt");
    cout << loadedTensor[0][0][150][458] << endl;
    at::Tensor pred = inference(imageTensor, model);
}

int main()
{
    cout << "Is CUDA available ? ";
    if (torch::cuda::is_available())
        cout << "Yes" << endl;
    else
    {
        cout << "No, end of the program" << endl;
        return -1;
    }

    string module_path = "E:\\twibot\\yolov7\\runs\\train\\yolov7-tiny-with-arm13\\weights\\traced_yolov7_model.pt";
    module_path = "E:\\twibot\\yolov7\\runs\\train\\yolov7-tiny-with-arm13\\weights\\best.torchscript.pt";
    torch::Device device(at::kCUDA);
    Module model = torch::jit::load(module_path, device);
    auto x = model.named_modules();
    warmup(model);

    processImage(model);

    return 0;
}
