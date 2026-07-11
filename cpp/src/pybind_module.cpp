#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "sightrag_core.hpp"

namespace py = pybind11;

py::array_t<uint8_t> mat_to_numpy(const cv::Mat& mat) {
    return py::array_t<uint8_t>(
        {mat.rows, mat.cols, mat.channels()},
        {(long)mat.step[0], (long)mat.step[1], sizeof(uint8_t)},
        mat.data
    );
}

PYBIND11_MODULE(sightrag_cpp, m) {
    m.doc() = "SightRAG C++ core — speed-critical operations";
    
    m.def("batch_load", [](const std::string& folder) {
        auto images = ImageOps::batch_load(folder);
        py::list result;
        for (auto& img : images) result.append(mat_to_numpy(img));
        return result;
    });
    
    m.def("batch_resize", [](py::list images, int w, int h) {
        std::vector<cv::Mat> mats;
        for (auto& item : images) {
            auto arr = item.cast<py::array_t<uint8_t>>();
            auto buf = arr.request();
            cv::Mat mat(buf.shape[0], buf.shape[1], CV_8UC3, buf.ptr);
            mats.push_back(mat);
        }
        auto resized = ImageOps::batch_resize(mats, w, h);
        py::list result;
        for (auto& img : resized) result.append(mat_to_numpy(img));
        return result;
    });
    
    m.def("extract_frames", [](const std::string& video, int fps) {
        auto frames = ImageOps::extract_frames(video, fps);
        py::list result;
        for (auto& f : frames) result.append(mat_to_numpy(f));
        return result;
    });
}
