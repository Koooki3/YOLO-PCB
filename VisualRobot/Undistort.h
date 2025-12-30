/**
 * @file Undistort.h
 * @brief 相机校准和图像去畸变模块头文件
 * 
 * 该文件定义了CameraCalibrator类，提供完整的相机校准和图像畸变校正功能，
 * 包括棋盘格角点提取、相机参数计算、图像去畸变和参数持久化等功能。
 * 
 * @author VisualRobot Team
 * @date 2025-12-30
 * @version 1.0
 */

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
 * 
 * @note 该类使用OpenCV的相机校准功能，支持从多张棋盘格图像中提取角点
 * @note 提供优化的图像去畸变算法，避免校正后出现黑边问题
 * @see prepareObjectPoints(), processImage(), calibrate(), undistortImage()
 */
class CameraCalibrator {
private:
    Size boardSize;          ///< 棋盘格内角点数量 (width, height)
    float squareSize;        ///< 棋盘格方格的实际大小 (单位: 毫米)
    vector<vector<Point3f>> objectPoints; ///< 世界坐标系中的三维点集合
    vector<vector<Point2f>> imagePoints;  ///< 图像坐标系中的二维点集合
    Size imageSize;          ///< 图像尺寸

    Mat cameraMatrix;        ///< 相机内参矩阵
    Mat distCoeffs;          ///< 畸变系数
    vector<Mat> rvecs, tvecs; ///< 旋转和平移向量集合

    double reprojectionError;   ///< 重投影误差

public:
    /**
     * @brief 构造函数
     * 
     * 初始化相机校准器，设置棋盘格参数
     * 
     * @param boardSize 棋盘格内角点数量 (width, height)
     * @param squareSize 棋盘格方格的实际大小 (单位: 毫米)，默认1.0f
     * @note boardSize参数决定了棋盘格的宽度和高度方向的内角点数量
     * @note squareSize用于生成世界坐标系中的三维点坐标
     */
    CameraCalibrator(Size boardSize, float squareSize = 1.0f);

    /**
     * @brief 准备物体点 (世界坐标系中的点)
     * 
     * 生成棋盘格在世界坐标系中的三维点，用于相机校准
     * 
     * @return 世界坐标系中的三维点集合
     * @note 世界坐标系以棋盘格左上角为原点，Z坐标为0
     * @note 点的排列顺序：按行优先，从左到右，从上到下
     * @see processImage(), calibrate()
     */
    vector<Point3f> prepareObjectPoints();

    /**
     * @brief 处理单张图像，提取角点
     * 
     * 从单张图像中提取棋盘格角点，并保存到图像点集合中
     * 
     * @param imagePath 图像文件路径
     * @param showResult 是否显示提取结果，默认false
     * @return 角点提取是否成功
     * @note 处理流程：
     *       1. 读取图像并转换为灰度图
     *       2. 使用findChessboardCorners检测棋盘格角点
     *       3. 使用cornerSubPix进行亚像素精确化
     *       4. 绘制角点标注并保存处理结果
     * @note 第一张图像会记录图像尺寸，后续图像会自动调整
     * @see processImagesFromFolder(), prepareObjectPoints()
     */
    bool processImage(const string& imagePath, bool showResult = false);

    /**
     * @brief 处理文件夹中的所有图像
     * 
     * 从文件夹中遍历所有图像，提取棋盘格角点，并保存到图像点集合中
     * 
     * @param folderPath 文件夹路径
     * @param showResult 是否显示提取结果，默认false
     * @return 成功处理的图像数量
     * @note 支持的图像格式：.jpg, .jpeg, .png, .bmp
     * @note 会自动跳过非图像文件和无法读取的图像
     * @see processImage()
     */
    int processImagesFromFolder(const string& folderPath, bool showResult = false);

    /**
     * @brief 执行相机校准
     * 
     * 使用提取的角点执行相机校准，计算相机内参和畸变系数
     * 
     * @return 重投影误差，如果失败返回-1
     * @note 校准要求：至少需要3张有效图像
     * @note 计算内容：
     *       - 相机内参矩阵 (3x3)
     *       - 畸变系数 (5x1)
     *       - 旋转和平移向量集合
     *       - 重投影误差
     * @note 使用CALIB_FIX_K3参数固定k3系数，减少数据需求
     * @see processImage(), saveCalibration()
     */
    double calibrate();

    /**
     * @brief 校正单张图像（优化版本，避免黑边问题）
     * 
     * 对输入的畸变图像进行校正，支持多种参数配置，避免校正后图像出现黑边问题
     * 
     * @param inputImage 输入的畸变图像
     * @param crop 是否裁剪到原始图像大小，默认true
     * @param borderSize 扩展边界的大小，默认为100像素
     * @param borderMode 边界填充模式，默认为BORDER_REPLICATE
     * @return 校正后的图像
     * @note 优化算法流程：
     *       1. 扩展输入图像边界，确保所有目标像素都能找到对应源像素
     *       2. 调整相机矩阵的光心坐标，考虑扩展的边界
     *       3. 使用getOptimalNewCameraMatrix优化新相机矩阵
     *       4. 生成映射数组并使用remap进行去畸变
     *       5. 根据crop参数决定是否裁剪回原始尺寸
     * @note 边界填充模式：
     *       - BORDER_REPLICATE: 复制边界像素
     *       - BORDER_CONSTANT: 填充常数
     *       - BORDER_REFLECT: 反射边界像素
     * @see calibrate()
     */
    Mat undistortImage(const Mat& inputImage, bool crop = true, int borderSize = 100, int borderMode = BORDER_REPLICATE);

    /**
     * @brief 保存校准参数到文件
     * 
     * 将相机内参、畸变系数等校准参数保存到文件中
     * 
     * @param filename 保存文件名
     * @return 保存是否成功
     * @note 保存内容：
     *       - camera_matrix: 相机内参矩阵
     *       - distortion_coefficients: 畸变系数
     *       - reprojection_error: 重投影误差
     *       - image_width: 图像宽度
     *       - image_height: 图像高度
     * @note 使用OpenCV的FileStorage格式（XML/YAML）
     * @see loadCalibration()
     */
    bool saveCalibration(const string& filename);

    /**
     * @brief 从文件加载校准参数
     * 
     * 从文件中加载相机内参、畸变系数等校准参数
     * 
     * @param filename 加载文件名
     * @return 加载是否成功
     * @note 加载内容与saveCalibration()保存的内容对应
     * @note 加载成功后可直接用于undistortImage()
     * @see saveCalibration()
     */
    bool loadCalibration(const string& filename);

    /**
     * @brief 获取相机内参矩阵
     * 
     * 获取当前校准得到的相机内参矩阵
     * 
     * @return 相机内参矩阵 (3x3)
     * @note 内参矩阵格式：
     *       [fx  0   cx]
     *       [0   fy  cy]
     *       [0   0   1 ]
     * @see getDistCoeffs()
     */
    Mat getCameraMatrix() const;
    
    /**
     * @brief 获取畸变系数
     * 
     * 获取当前校准得到的畸变系数向量
     * 
     * @return 畸变系数 (5x1)
     * @note 畸变系数顺序：[k1, k2, p1, p2, k3]
     *       - k1, k2, k3: 径向畸变系数
     *       - p1, p2: 切向畸变系数
     * @see getCameraMatrix()
     */
    Mat getDistCoeffs() const;
    
    /**
     * @brief 获取重投影误差
     * 
     * 获取相机校准的重投影误差
     * 
     * @return 重投影误差
     * @note 重投影误差越小，校准精度越高
     * @note 通常误差小于0.5像素表示良好的校准结果
     */
    double getReprojectionError() const;
};

#endif // UNDISTORT_H
