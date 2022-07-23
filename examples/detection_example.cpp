#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include "feature_lib.h"

std::vector<uint8_t> matToVector(const cv::Mat &mat) {
  std::vector<uint8_t> result;
  cv::Mat cont = mat.isContinuous() ? mat : mat.clone();  // make sure the Mat is continuous
  cont.reshape(1, 1).copyTo(result);  // copy from a 1D Mat
  return result;
}

void printOutput(const featurelib::DetectionResult &output){
  std::cout << "Printing points" << std::endl;
  for (auto &kp : output.keypoints) {
    std::cout << kp.x << "\t" << kp.y << "\t" << kp.size << "\t" << kp.angle << "\t" << kp.strength << std::endl;
  }

  std::cout << "Printing descriptors" << std::endl;
  for (const auto &row : output.descriptors) {
    for (const auto &element : row) std::cout << element << ' ';
    std::cout << std::endl;
  }
}

// Calculates key points and image descriptors
// Takes an image file path as input
// Prints key points and image descriptors to stdout
int main(int argc, char *argv[]) {
  if (argc != 2) {
    std::cerr << "Illegal CLI arguments num. Expected only a file path but was " << argc
              << " arguments (including the command name)" << std::endl;
    return 1;
  }
  auto file_path = argv[1];
  cv::Mat img = imread(file_path, cv::IMREAD_COLOR);

  auto width = img.cols;
  auto height = img.rows;

  std::vector<uint8_t> image_data = matToVector(img);

  featurelib::SiftDetector sift_detector(width, height);
  featurelib::DetectionResult sift_output = sift_detector.detect(image_data);

  featurelib::OrbDetector orb_detector(width, height);
  featurelib::DetectionResult orb_output = orb_detector.detect(image_data);

  featurelib::SurfDetector surf_detector(width, height);
  featurelib::DetectionResult surf_output = surf_detector.detect(image_data);

  printOutput(sift_output);
  printOutput(orb_output);
  printOutput(surf_output);
  return 0;
}
