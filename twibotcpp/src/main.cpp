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

template<typename T>
std::vector<size_t> argsort(const std::vector<T> &array) {
    std::vector<size_t> indices(array.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(),
              [&array](int left, int right) -> bool {
                  // sort indices according to corresponding array element
                  return array[left] < array[right];
              });

    return indices;
}

int chooseColor(const cv::Mat& card) {
    // Step 2: Convert the image to the HSV color space
    cv::Mat hsvImage;
    cvtColor(card, hsvImage, cv::COLOR_BGR2HSV);

    // Step 3: Calculate the dominant color
    // Prepare a histogram of the hue component
    int hBins = 256; // Number of bins for hue (0-255)
    int histSize[] = { hBins };
    float hRanges[] = { 0, 256 };
    const float* ranges[] = { hRanges };
    cv::MatND hist;
    int channels[] = { 0 }; // We are interested in the 0-th channel (hue)

    calcHist(&hsvImage, 1, channels, cv::Mat(), hist, 1, histSize, ranges, true, false);

    // Find the index of the most frequent hue value
    double maxVal = 0;
    int dominantHueIndex = 0;
    for (int i = 1; i < hBins; ++i) {
        double binValue = hist.at<float>(i);
        if (binValue > maxVal) {
            maxVal = binValue;
            dominantHueIndex = i;
        }
    }

    // The dominant color's hue value
    return dominantHueIndex;
}


cv::Scalar ScalarHSV2BGR(uchar H, uchar S, uchar V) {
    cv::Mat rgb;
    cv::Mat hsv(1,1, CV_8UC3, cv::Scalar(H,S,V));
    cv::cvtColor(hsv, rgb, cv::COLOR_HSV2BGR);
    return cv::Scalar(rgb.data[0], rgb.data[1], rgb.data[2]);
}

cv::Mat& drawDetections(cv::Mat& img,
        const DetectionMatrix & detections,
        const std::vector<pair_t>* pairs)
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

    if (!pairs->empty()) {
        // choose color
        std::vector<int> xCoordsToSort;
        for (auto pair : *pairs) {
            int xCard1 = detections[0][pair._1].bbox.x;
            int xCard2 = detections[0][pair._2].bbox.x;
            xCoordsToSort.push_back(std::max(xCard1, xCard2));
        }

        std::vector<size_t> indicesSorted = argsort(xCoordsToSort);

        int countColor = 0;
        for (auto pi: indicesSorted) {
            const auto &box1 = detections[0][(*pairs)[pi]._1].bbox;
            const auto &box2 = detections[0][(*pairs)[pi]._2].bbox;

            /*uchar hue = (uchar) chooseColor(img(box1));
            std::cout << +hue << std::endl;*/

            cv::rectangle(img, box1, colors_hex[countColor], 2);
            cv::rectangle(img, box2, colors_hex[countColor], 2);
            countColor++;
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

void Demo(cv::Mat& img, const DetectionMatrix & detections, const std::vector<pair_t> & pairs) {

    img = drawDetections(img, detections, &pairs);

    cv::namedWindow("Result", cv::WINDOW_AUTOSIZE);
    cv::imshow("Result", img);
    cv::waitKey(0);
}


std::vector<pair_t> matchCards(DetectionMatrix& detectedCards, const cv::Ptr<cv::SiftFeatureDetector>& sift, cv::Mat& frame, const cv::Ptr<cv::DescriptorMatcher>& matcher) {
    std::vector<Detection> cardResults {};
    std::vector<pair_t> pairs {};
    unsigned long nbCards = detectedCards.size();
    if (nbCards > 30 || nbCards == 0) return pairs;
    for (const auto& r: detectedCards[0]) {
        if (r.class_idx == 0)
            cardResults.push_back(r);
    }
    nbCards = cardResults.size();
    if (nbCards < 2) return pairs;
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

            if (good_matches > 15) {
                pairs.push_back(pair_t{i, j});
            }
        }
    }
    return pairs;
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

    if (ext == ".jpg" || ext == ".png") {
        cv::Mat img = cv::imread(source);
        if (img.empty()) {
            std::cerr << "Error loading the image " << source << std::endl;
            return -1;
        }

        // inference
        DetectionMatrix result = detector.Run(img, conf_thres, iou_thres);
        std::vector<pair_t> pairsFound = matchCards(result, sift, img, matcher);

        // visualize detections
        if (opt["view-img"].as<bool>()) {
            Demo(img, result, pairsFound);
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
        std::vector<pair_t>* currentPairs = nullptr;
        std::vector<pair_t> pairsFound;
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
                // cv::imwrite("../debug/colordebug/frame-" + std::to_string(count) + ".png", frame);
                if (currentPairs != nullptr) currentPairs->clear();
                currentResults = detector.Run(frame, conf_thres, iou_thres);
                pairsFound = matchCards(currentResults, sift, frame, matcher);
                currentPairs = &pairsFound;
            }
            if (currentPairs == nullptr || currentPairs->empty()) continue;
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