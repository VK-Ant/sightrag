#include "sightrag_core.hpp"
#include <algorithm>

std::vector<cv::Mat> ImageOps::batch_load(const std::string& folder) {
    std::vector<std::string> paths;
    for (const auto& entry : fs::directory_iterator(folder)) {
        std::string ext = entry.path().extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        if (ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".webp") {
            paths.push_back(entry.path().string());
        }
    }
    std::sort(paths.begin(), paths.end());
    std::vector<cv::Mat> images;
    images.reserve(paths.size());
    for (const auto& p : paths) {
        cv::Mat img = cv::imread(p, cv::IMREAD_COLOR);
        if (!img.empty()) images.push_back(img);
    }
    return images;
}

std::vector<cv::Mat> ImageOps::batch_resize(const std::vector<cv::Mat>& images, int w, int h) {
    std::vector<cv::Mat> resized;
    resized.reserve(images.size());
    for (const auto& img : images) {
        cv::Mat r;
        cv::resize(img, r, cv::Size(w, h), 0, 0, cv::INTER_LINEAR);
        resized.push_back(r);
    }
    return resized;
}

std::vector<cv::Mat> ImageOps::batch_crop(const cv::Mat& image, const std::vector<cv::Rect>& regions) {
    std::vector<cv::Mat> crops;
    crops.reserve(regions.size());
    for (const auto& roi : regions) {
        cv::Rect safe(std::max(0, roi.x), std::max(0, roi.y),
                      std::min(roi.width, image.cols - std::max(0, roi.x)),
                      std::min(roi.height, image.rows - std::max(0, roi.y)));
        if (safe.width > 0 && safe.height > 0)
            crops.push_back(image(safe).clone());
    }
    return crops;
}

std::vector<cv::Mat> ImageOps::extract_frames(const std::string& video_path, int target_fps) {
    cv::VideoCapture cap(video_path);
    if (!cap.isOpened()) throw std::runtime_error("Cannot open: " + video_path);
    double fps = cap.get(cv::CAP_PROP_FPS);
    int interval = std::max(1, static_cast<int>(fps / target_fps));
    std::vector<cv::Mat> frames;
    cv::Mat frame;
    int count = 0;
    while (cap.read(frame)) {
        if (count % interval == 0) frames.push_back(frame.clone());
        count++;
    }
    cap.release();
    return frames;
}
