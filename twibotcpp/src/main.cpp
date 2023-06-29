#include <iostream>
#include <memory>
#include <chrono>
#include <cstring>

#include <opencv2/features2d.hpp>
#include <opencv2/opencv.hpp>

#include "detector.h"
#include "cxxopts.hpp"

typedef struct pair_s {
    int _1;
    int _2;
} pair_t;

using DetectionMatrix = std::vector<std::vector<Detection>>;


std::vector<std::string> LoadNames(const std::string& path) {
    // load class names
    std::vector<std::string> class_names;
    std::ifstream infile(path);
    if (infile.is_open()) {
        std::string line;
        while (getline (infile,line)) {
            class_names.emplace_back(line);
        }
        infile.close();
    }
    else {
        std::cerr << "Error loading the class names!\n";
    }

    return class_names;
}


cv::Mat& drawDetections(cv::Mat& img,
                        const DetectionMatrix & detections)
{
    if (!detections.empty()) {
        for (const auto& detection : detections[0]) {
            int class_idx = detection.class_idx;
            if (class_idx == 1) break;

            const auto &box = detection.bbox;

            cv::rectangle(img, box, cv::Scalar(0, 0, 255), 2);
        }
    }
    return img;
}

// TODO !!
cv::Mat& drawDetections(cv::Mat& img,
        const DetectionMatrix & detections,
        const std::vector<pair_t>& pairs)
{
    cv::Scalar colors_hex[10]{
            cv::Scalar(0xc7, 0x15, 0x85),
            cv::Scalar(0x00, 0x00, 0xcd),
            cv::Scalar(0x00, 0xff, 0x00),
            cv::Scalar(0xff, 0xff, 0x00),
            cv::Scalar(0xff, 0x45, 0x00),
            cv::Scalar(0x2f, 0x4f, 0x4f),
            cv::Scalar(0x22, 0x8b, 0x22),
            cv::Scalar(0x00, 0xff, 0xff),
            cv::Scalar(0x1e, 0x90, 0xff),
            cv::Scalar(0xff, 0xde, 0xad)
    };

    if (!pairs.empty()) {
        for (std::size_t pi = 0; pi < pairs.size(); pi++) {
            const auto &box1 = detections[0][pairs[pi]._1].bbox;
            const auto &box2 = detections[0][pairs[pi]._2].bbox;

            cv::rectangle(img, box1, colors_hex[pi], 2);
            cv::rectangle(img, box2, colors_hex[pi], 2);
        }
    }
    return img;
}

void Demo(cv::Mat& img, const DetectionMatrix & detections) {

    img = drawDetections(img, detections);

    cv::namedWindow("Result", cv::WINDOW_AUTOSIZE);
    cv::imshow("Result", img);
    cv::waitKey(0);
}


int main(int argc, const char* argv[]) {
    cxxopts::Options parser(argv[0], "A LibTorch inference implementation of the yolov5");

    // TODO: add other args
    parser.allow_unrecognised_options().add_options()
            ("weights", "model.torchscript.pt path", cxxopts::value<std::string>())
            ("source", "source", cxxopts::value<std::string>())
            ("output", "output", cxxopts::value<std::string>()->default_value("output.avi"))
            ("conf-thres", "object confidence threshold", cxxopts::value<float>()->default_value("0.4"))
            ("iou-thres", "IOU threshold for NMS", cxxopts::value<float>()->default_value("0.5"))
            ("gpu", "Enable cuda device or cpu", cxxopts::value<bool>()->default_value("false"))
            ("view-img", "display results", cxxopts::value<bool>()->default_value("false"))
            ("h,help", "Print usage");

    auto opt = parser.parse(argc, argv);

    if (opt.count("help")) {
        std::cout << parser.help() << std::endl;
        exit(0);
    }

    // check if gpu flag is set
    bool is_gpu = opt["gpu"].as<bool>();

    // set device type - CPU/GPU
    torch::DeviceType device_type;
    if (torch::cuda::is_available() && is_gpu) {
        device_type = torch::kCUDA;
    } else {
        device_type = torch::kCPU;
    }

    // load class names from dataset for visualization
    std::vector<std::string> class_names = LoadNames("../weights/card.names");
    if (class_names.empty()) {
        return -1;
    }

    // load network
    std::string weights = opt["weights"].as<std::string>();
    Detector detector = Detector(weights, device_type);

    // run once to warm up
    std::cout << "Run once on empty image" << std::endl;
    auto temp_img = cv::Mat::zeros(720, 1280, CV_32FC3);
    detector.Run(temp_img, 1.0f, 1.0f);

    // set up threshold
    float conf_thres = opt["conf-thres"].as<float>();
    float iou_thres = opt["iou-thres"].as<float>();

    // load input source
    std::string source = opt["source"].as<std::string>();
    std::string ext = source.substr(source.length() - 4);

    // load sift detector
    cv::Ptr<cv::SiftFeatureDetector> sift = cv::SiftFeatureDetector::create();
    std::vector<cv::KeyPoint> keypoints;

    // load matcher
    cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);

    if (ext == ".jpg") {
        cv::Mat img = cv::imread(source);
        if (img.empty()) {
            std::cerr << "Error loading the image " << source << std::endl;
            return -1;
        }

        // inference
        DetectionMatrix result = detector.Run(img, conf_thres, iou_thres);

        // visualize detections
        if (opt["view-img"].as<bool>()) {
            Demo(img, result);
        }
    }
    else if (ext == ".mp4" || ext == ".MOV" || ext == ".avi"){
        cv::VideoCapture video(source);
        std::cout << video.isOpened() << std::endl;
        if (!video.isOpened()) {
            std::cerr << "Error loading the video " << source << std::endl;
            return EXIT_FAILURE;
        }

        // load VideoWriter output
        std::string output_path = opt["output"].as<std::string>();
        cv::VideoWriter videoOutput(output_path, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 50, cv::Size(1280, 720));

        int count = 0;
        std::vector<pair_t> currentPairs;
        DetectionMatrix currentResults;
        while (true) {
            count++;
            std::cout << count << std::endl;

            cv::Mat frame;
            video >> frame;

            if (frame.empty()) {
                std::cerr << "Video reach an empty frame" << std::endl;
                break;
            }

            if (count % 5 == 4) {
                currentPairs.clear();
                currentResults = detector.Run(frame, conf_thres, iou_thres);
                std::vector<Detection> cardResults {};
                unsigned long nbCards = currentResults.size();
                if (nbCards > 30 || nbCards == 0) continue;
                for (const auto& r: currentResults[0]) {
                    if (r.class_idx == 0)
                        cardResults.push_back(r);
                }
                nbCards = cardResults.size();
                if (nbCards == 0) continue;
                bool *alreadyComputed{ new bool[nbCards]{false} };
                std::vector<cv::KeyPoint> *kps{ new std::vector<cv::KeyPoint>[nbCards]};
                cv::Mat *dss{ new cv::Mat[nbCards]};

                for (int i = 0; i < nbCards; i++) {
                    std::vector<cv::KeyPoint> kpi;
                    cv::Mat dsi;
                    if (!alreadyComputed[i]) {
                        sift->detectAndCompute(frame(cardResults[i].bbox), cv::Mat(), kpi, dsi);
                        alreadyComputed[i] = true;
                        kps[i] = kpi;
                        dss[i] = dsi;
                    }
                    else {
                        kpi = kps[i];
                        dsi = dss[i];
                    }
                    for (int j = i+1; j < nbCards; j++) {
                        std::vector<cv::KeyPoint> kpj;
                        cv::Mat dsj;

                        if (!alreadyComputed[j]) {
                            sift->detectAndCompute(frame(cardResults[j].bbox), cv::Mat(), kpj, dsj);
                            alreadyComputed[j] = true;
                            kps[j] = kpj;
                            dss[j] = dsj;
                        }
                        else {
                            kpj = kps[j];
                            dsj = dss[j];
                        }

                        std::vector< std::vector<cv::DMatch> > knn_matches;

                        if (dsi.data == nullptr || dsj.data == nullptr) {
                            std::cout << "nullptr" << std::endl;
                            continue;
                        }

                        if (kpi.size() < 2 || kpj.size() < 2) {
                            std::cout << "size" << std::endl;
                            continue;
                        }

                        matcher->knnMatch( dsi, dsj, knn_matches, 2 );

                        const float ratio_thresh = 0.4f;
                        int good_matches = 0;
                        for (auto & knn_match_ : knn_matches)
                            if (knn_match_[0].distance < ratio_thresh * knn_match_[1].distance)
                                good_matches++;

                        if (good_matches > 5) {
                            currentPairs.push_back(pair_t{i, j});
                        }
                    }
                }
            }
            cv::Mat resultWithDet = drawDetections(frame, currentResults, currentPairs);
            videoOutput.write(resultWithDet);
        }

        video.release();
        videoOutput.release();
    }
    else {
        throw std::invalid_argument("Argument " + ext + " is invalid");
    }

    cv::destroyAllWindows();
    return EXIT_SUCCESS;
}