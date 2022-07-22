#ifndef FEATUREDETECT_FEATURELIB_H_
#define FEATUREDETECT_FEATURELIB_H_

#include <vector>
#include <opencv2/features2d.hpp>

namespace featurelib {

struct KeyPoint {
  float x;
  float y;
  float size;
  float angle;
  float strength;
};

struct DetectionResult {
  std::vector<KeyPoint> keypoints;
  std::vector<std::vector<uint8_t>> descriptors;
};

class FeatureDetector {
 public:
  virtual DetectionResult detect(const std::vector<std::uint8_t> &input_vector) const = 0;

  void setHeight(int value) {
    height = value;
    mask = cv::Mat(255 * cv::Mat::ones(height, width, CV_8U));
  }

  void setWidth(int value) {
    width = value;
    mask = cv::Mat(255 * cv::Mat::ones(height, width, CV_8U));
  }

  int getWidth() const {
    return width;
  }

  int getHeight() const {
    return height;
  }

 protected:
  int width = 0;
  int height = 0;
  cv::Mat mask = cv::Mat(255 * cv::Mat::ones(height, width, CV_8U));
};

class SiftDetector : public FeatureDetector {
 public:
  SiftDetector();

  SiftDetector(int width, int height);

  DetectionResult detect(const std::vector<std::uint8_t> &input) const override;

 private:
  cv::Ptr<cv::Feature2D> sift_ = cv::SIFT::create();
};

} // namespace featurelib

#endif // FEATUREDETECT_FEATURELIB_H_
