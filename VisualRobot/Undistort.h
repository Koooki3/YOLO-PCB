#ifndef UNDISTORT_H
#define UNDISTORT_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <filesystem>

using namespace std;
using namespace cv;

namespace fs = filesystem;

/**
 * @brief 相机校准器类
 * 
 * 用于相机校准和图像畸变校正，支持从棋盘格图像中提取角点，计算相机内参和畸变系数，
 * 并提供图像畸变校正功能
 */
class CameraCalibrator {
private:
    Size boardSize;          // 棋盘格内角点数量 (width, height)
    float squareSize;        // 棋盘格方格的实际大小 (单位: 毫米) 
    vector<vector<Point3f>> objectPoints; // 世界坐标系中的三维点集合
    vector<vector<Point2f>> imagePoints;  // 图像坐标系中的二维点集合
    Size imageSize;          // 图像尺寸

    Mat cameraMatrix;        // 相机内参矩阵
    Mat distCoeffs;          // 畸变系数
    vector<Mat> rvecs, tvecs; // 旋转和平移向量集合

    double reprojectionError;   // 重投影误差

public:
    /**
     * @brief 构造函数
     * @param boardSize 棋盘格内角点数量 (width, height)
     * @param squareSize 棋盘格方格的实际大小 (单位: 毫米)，默认1.0f
     */
    CameraCalibrator(Size boardSize, float squareSize = 1.0f);

    /**
     * @brief 准备物体点 (世界坐标系中的点)
     * @return 世界坐标系中的三维点集合
     * 
     * 生成棋盘格在世界坐标系中的三维点，用于相机校准
     */
    vector<Point3f> prepareObjectPoints();

    /**
     * @brief 处理单张图像，提取角点
     * @param imagePath 图像文件路径
     * @param showResult 是否显示提取结果，默认false
     * @return 角点提取是否成功
     * 
     * 从单张图像中提取棋盘格角点，并保存到图像点集合中
     */
    bool processImage(const string& imagePath, bool showResult = false);

    /**
     * @brief 处理文件夹中的所有图像
     * @param folderPath 文件夹路径
     * @param showResult 是否显示提取结果，默认false
     * @return 成功处理的图像数量
     * 
     * 从文件夹中遍历所有图像，提取棋盘格角点，并保存到图像点集合中
     */
    int processImagesFromFolder(const string& folderPath, bool showResult = false);

    /**
     * @brief 执行相机校准
     * @return 重投影误差
     * 
     * 使用提取的角点执行相机校准，计算相机内参和畸变系数
     */
    double calibrate();

    /**
     * @brief 校正单张图像（优化版本，避免黑边问题）
     * @param inputImage 输入的畸变图像
     * @param crop 是否裁剪到原始图像大小，默认true
     * @param borderSize 扩展边界的大小，默认为100像素
     * @param borderMode 边界填充模式，默认为BORDER_REPLICATE
     * @return 校正后的图像
     * 
     * 对输入的畸变图像进行校正，支持多种参数配置，避免校正后图像出现黑边问题
     */
    Mat undistortImage(const Mat& inputImage, bool crop = true, int borderSize = 100, int borderMode = BORDER_REPLICATE);

    /**
     * @brief 保存校准参数到文件
     * @param filename 保存文件名
     * @return 保存是否成功
     * 
     * 将相机内参、畸变系数等校准参数保存到文件中
     */
    bool saveCalibration(const string& filename);

    /**
     * @brief 从文件加载校准参数
     * @param filename 加载文件名
     * @return 加载是否成功
     * 
     * 从文件中加载相机内参、畸变系数等校准参数
     */
    bool loadCalibration(const string& filename);

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
