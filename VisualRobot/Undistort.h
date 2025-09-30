#ifndef UNDISTORT_H
#define UNDISTORT_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <filesystem>

namespace fs = std::filesystem;

class CameraCalibrator {
private:
    cv::Size boardSize;          // 棋盘格内角点数量 (width, height)
    float squareSize;           // 棋盘格方格的实际大小 (单位: 毫米) 
    std::vector<std::vector<cv::Point3f>> objectPoints; // 世界坐标系中的三维点
    std::vector<std::vector<cv::Point2f>> imagePoints;  // 图像坐标系中的二维点
    cv::Size imageSize;         // 图像尺寸

    cv::Mat cameraMatrix;       // 相机内参矩阵
    cv::Mat distCoeffs;         // 畸变系数
    std::vector<cv::Mat> rvecs, tvecs; // 旋转和平移向量

    double reprojectionError;   // 重投影误差

public:
    CameraCalibrator(cv::Size boardSize, float squareSize = 1.0f);

    // 准备物体点 (世界坐标系中的点) 
    std::vector<cv::Point3f> prepareObjectPoints();

    // 处理单张图像，提取角点
    bool processImage(const std::string& imagePath, bool showResult = false);

    // 处理文件夹中的所有图像
    int processImagesFromFolder(const std::string& folderPath, bool showResult = false);

    // 执行相机校准
    double calibrate();

    // 校正单张图像
    cv::Mat undistortImage(const cv::Mat& inputImage, bool crop = true);

    // 保存校准参数到文件
    bool saveCalibration(const std::string& filename);

    // 从文件加载校准参数
    bool loadCalibration(const std::string& filename);

    // 获取相机参数
    cv::Mat getCameraMatrix() const;
    cv::Mat getDistCoeffs() const;
    double getReprojectionError() const;
};

#endif // UNDISTORT_H
