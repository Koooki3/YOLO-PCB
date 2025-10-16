#ifndef UNDISTORT_H
#define UNDISTORT_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <filesystem>

/**
 * @file Undistort.h
 * @brief 相机标定和图像校正类头文件
 * @author VisualRobot Team
 * @date 2025
 */

namespace fs = std::filesystem;
using namespace cv;

/**
 * @class CameraCalibrator
 * @brief 相机标定和图像校正类
 * 
 * 该类提供相机标定、畸变校正和参数保存/加载功能
 */
class CameraCalibrator {
private:
    Size boardSize_;           ///< 棋盘格内角点数量 (width, height)
    float squareSize_;         ///< 棋盘格方格的实际大小 (单位: 毫米)
    std::vector<std::vector<Point3f>> objectPoints_; ///< 世界坐标系中的三维点
    std::vector<std::vector<Point2f>> imagePoints_;  ///< 图像坐标系中的二维点
    Size imageSize_;           ///< 图像尺寸

    Mat cameraMatrix_;         ///< 相机内参矩阵
    Mat distCoeffs_;           ///< 畸变系数
    std::vector<Mat> rvecs_;   ///< 旋转向量
    std::vector<Mat> tvecs_;   ///< 平移向量

    double reprojectionError_; ///< 重投影误差

public:
    /**
     * @brief 构造函数
     * @param boardSize 棋盘格内角点数量
     * @param squareSize 棋盘格方格的实际大小 (默认1.0毫米)
     */
    CameraCalibrator(Size boardSize, float squareSize = 1.0f);

    /**
     * @brief 准备物体点 (世界坐标系中的点)
     * @return 世界坐标系中的三维点集合
     */
    std::vector<Point3f> prepareObjectPoints();

    /**
     * @brief 处理单张图像，提取角点
     * @param imagePath 图像文件路径
     * @param showResult 是否显示处理结果 (默认false)
     * @return 成功返回true，失败返回false
     */
    bool processImage(const std::string& imagePath, bool showResult = false);

    /**
     * @brief 处理文件夹中的所有图像
     * @param folderPath 文件夹路径
     * @param showResult 是否显示处理结果 (默认false)
     * @return 成功处理的图像数量
     */
    int processImagesFromFolder(const std::string& folderPath, bool showResult = false);

    /**
     * @brief 执行相机校准
     * @return 重投影误差，失败返回-1
     */
    double calibrate();

    /**
     * @brief 校正单张图像
     * @param inputImage 输入图像
     * @param crop 是否裁剪黑边 (默认true)
     * @return 校正后的图像
     */
    Mat undistortImage(const Mat& inputImage, bool crop = true);

    /**
     * @brief 保存校准参数到文件
     * @param filename 文件名
     * @return 成功返回true，失败返回false
     */
    bool saveCalibration(const std::string& filename);

    /**
     * @brief 从文件加载校准参数
     * @param filename 文件名
     * @return 成功返回true，失败返回false
     */
    bool loadCalibration(const std::string& filename);

    /**
     * @brief 获取相机内参矩阵
     * @return 相机内参矩阵
     */
    Mat getCameraMatrix() const;

    /**
     * @brief 获取畸变系数
     * @return 畸变系数
     */
    Mat getDistCoeffs() const;

    /**
     * @brief 获取重投影误差
     * @return 重投影误差
     */
    double getReprojectionError() const;
};

#endif // UNDISTORT_H
