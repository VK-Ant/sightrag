#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <filesystem>

namespace fs = std::filesystem;

class ImageOps {
public:
    static std::vector<cv::Mat> batch_load(const std::string& folder);
    static std::vector<cv::Mat> batch_resize(const std::vector<cv::Mat>& images, int w = 224, int h = 224);
    static std::vector<cv::Mat> batch_crop(const cv::Mat& image, const std::vector<cv::Rect>& regions);
    static std::vector<cv::Mat> extract_frames(const std::string& video_path, int target_fps = 1);
};
