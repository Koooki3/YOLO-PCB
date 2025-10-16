/**
 * @file Undistort.cpp
 * @brief 相机标定和图像校正类实现
 * @author VisualRobot Team
 * @date 2025
 */

#include "Undistort.h"
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace cv;

/**
 * @brief 构造函数
 * @param boardSize 棋盘格内角点数量
 * @param squareSize 棋盘格方格的实际大小 (默认1.0毫米)
 */
CameraCalibrator::CameraCalibrator(Size boardSize, float squareSize)
    : boardSize_(boardSize), squareSize_(squareSize) {}

/**
 * @brief 准备物体点 (世界坐标系中的点)
 * @return 世界坐标系中的三维点集合
 */
std::vector<Point3f> CameraCalibrator::prepareObjectPoints()
{
    std::vector<Point3f> objPoints;
    for (int i = 0; i < boardSize_.height; i++)
    {
        for (int j = 0; j < boardSize_.width; j++)
        {
            objPoints.push_back(Point3f(j * squareSize_, i * squareSize_, 0.0f));
        }
    }
    return objPoints;
}

/**
 * @brief 处理单张图像，提取角点
 * @param imagePath 图像文件路径
 * @param showResult 是否显示处理结果 (默认false)
 * @return 成功返回true，失败返回false
 */
bool CameraCalibrator::processImage(const std::string& imagePath, bool showResult)
{
    Mat image = imread(imagePath);
    if (image.empty())
    {
        std::cerr << "无法读取图像: " << imagePath << std::endl;
        return false;
    }

    // 如果是第一张图像，记录图像尺寸
    if (imageSize_.empty())
    {
        imageSize_ = image.size();
    }

    Mat gray;
    cvtColor(image, gray, COLOR_BGR2GRAY);

    std::vector<Point2f> corners;
    bool found = findChessboardCorners(gray, boardSize_, corners);

    if (found)
    {
        // 亚像素精确化
        TermCriteria criteria(TermCriteria::EPS + TermCriteria::MAX_ITER, 30, 0.001);
        cornerSubPix(gray, corners, Size(11, 11), Size(-1, -1), criteria);

        // 保存角点
        imagePoints_.push_back(corners);
        std::vector<Point3f> objPoints = prepareObjectPoints();
        objectPoints_.push_back(objPoints);

        // 创建带标注的图像
        Mat annotatedImage = image.clone();
        drawChessboardCorners(annotatedImage, boardSize_, corners, found);

        // 在图像上标注每个角点的坐标
        for (size_t i = 0; i < corners.size(); i++)
        {
            Point2f pixelCoord = corners[i];
            Point3f worldCoord = objPoints[i];

            // 创建坐标文本
            std::string pixelText = "Pix: (" + std::to_string((double)pixelCoord.x) + ", " + std::to_string((double)pixelCoord.y) + ")";
            std::string worldText = "World: (" + std::to_string(worldCoord.x) + ", " + std::to_string(worldCoord.y) + ", " + std::to_string(worldCoord.z) + ")";

            // 设置文本位置（稍微偏移以避免重叠）
            Point textPos(pixelCoord.x + 10, pixelCoord.y - 10);

            // 绘制像素坐标（绿色）
            putText(annotatedImage, pixelText, textPos, FONT_HERSHEY_SIMPLEX, 0.4, Scalar(0, 255, 0), 1);

            // 绘制世界坐标（黄色）
            putText(annotatedImage, worldText, Point(textPos.x, textPos.y + 15), FONT_HERSHEY_SIMPLEX, 0.4, Scalar(0, 255, 255), 1);
        }

        // 保存处理后的图像
        fs::path pathObj(imagePath);
        std::string processedPath = pathObj.parent_path().string() + "/" + pathObj.stem().string() + "_processed" + pathObj.extension().string();
        imwrite(processedPath, annotatedImage);
        std::cout << "已保存处理结果: " << processedPath << std::endl;

        if (showResult)
        {
            imshow("角点检测", annotatedImage);
            waitKey(0);
        }

        return true;
    }
    else
    {
        std::cerr << "未找到棋盘格角点: " << imagePath << std::endl;
        return false;
    }
}

/**
 * @brief 处理文件夹中的所有图像
 * @param folderPath 文件夹路径
 * @param showResult 是否显示处理结果 (默认false)
 * @return 成功处理的图像数量
 */
int CameraCalibrator::processImagesFromFolder(const std::string& folderPath, bool showResult)
{
    int processedCount = 0;

    try
    {
        for (const auto& entry : fs::directory_iterator(folderPath))
        {
            if (entry.is_regular_file())
            {
                std::string ext = entry.path().extension().string();
                // 检查常见图像格式
                if (ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp")
                {
                    if (processImage(entry.path().string(), showResult))
                    {
                        processedCount++;
                    }
                }
            }
        }
    }
    catch (const fs::filesystem_error& e)
    {
        std::cerr << "文件系统错误: " << e.what() << std::endl;
    }

    std::cout << "成功处理 " << processedCount << " 张图像" << std::endl;
    return processedCount;
}

/**
 * @brief 执行相机校准
 * @return 重投影误差，失败返回-1
 */
double CameraCalibrator::calibrate()
{
    if (imagePoints_.size() < 10)
    {
        std::cerr << "需要至少10张有效图像进行校准，当前只有 " << imagePoints_.size() << " 张" << std::endl;
        return -1;
    }

    std::vector<Mat> rvecs, tvecs;
    reprojectionError_ = calibrateCamera(
        objectPoints_, imagePoints_, imageSize_,
        cameraMatrix_, distCoeffs_, rvecs, tvecs,
        CALIB_FIX_K3 // 固定k3系数，通常k3影响不大且需要更多数据
    );

    std::cout << "重投影误差: " << reprojectionError_ << std::endl;
    std::cout << "相机内参矩阵:\n" << cameraMatrix_ << std::endl;
    std::cout << "畸变系数: " << distCoeffs_.t() << std::endl;

    return reprojectionError_;
}

/**
 * @brief 校正单张图像
 * @param inputImage 输入图像
 * @param crop 是否裁剪黑边 (默认true)
 * @return 校正后的图像
 */
Mat CameraCalibrator::undistortImage(const Mat& inputImage, bool crop)
{
    Mat undistorted;

    // 获取优化后的新相机矩阵
    Mat newCameraMatrix = getOptimalNewCameraMatrix(cameraMatrix_, distCoeffs_, imageSize_, 1.0, imageSize_);

    // 校正图像
    undistort(inputImage, undistorted, cameraMatrix_, distCoeffs_, newCameraMatrix);

    // 如果需要，裁剪黑边
    if (crop)
    {
        Rect roi;
        getOptimalNewCameraMatrix(cameraMatrix_, distCoeffs_, imageSize_, 0, imageSize_, &roi);
        undistorted = undistorted(roi);
    }

    return undistorted;
}

/**
 * @brief 保存校准参数到文件
 * @param filename 文件名
 * @return 成功返回true，失败返回false
 */
bool CameraCalibrator::saveCalibration(const std::string& filename)
{
    FileStorage fs(filename, FileStorage::WRITE);
    if (!fs.isOpened())
    {
        std::cerr << "无法创建文件: " << filename << std::endl;
        return false;
    }

    fs << "camera_matrix" << cameraMatrix_;
    fs << "distortion_coefficients" << distCoeffs_;
    fs << "reprojection_error" << reprojectionError_;
    fs << "image_width" << imageSize_.width;
    fs << "image_height" << imageSize_.height;

    fs.release();
    std::cout << "校准参数已保存到: " << filename << std::endl;
    return true;
}

/**
 * @brief 从文件加载校准参数
 * @param filename 文件名
 * @return 成功返回true，失败返回false
 */
bool CameraCalibrator::loadCalibration(const std::string& filename)
{
    FileStorage fs(filename, FileStorage::READ);
    if (!fs.isOpened())
    {
        std::cerr << "无法打开文件: " << filename << std::endl;
        return false;
    }

    fs["camera_matrix"] >> cameraMatrix_;
    fs["distortion_coefficients"] >> distCoeffs_;
    fs["reprojection_error"] >> reprojectionError_;

    int width, height;
    fs["image_width"] >> width;
    fs["image_height"] >> height;
    imageSize_ = Size(width, height);

    fs.release();
    std::cout << "校准参数已从 " << filename << " 加载" << std::endl;
    return true;
}

/**
 * @brief 获取相机内参矩阵
 * @return 相机内参矩阵
 */
Mat CameraCalibrator::getCameraMatrix() const { return cameraMatrix_; }

/**
 * @brief 获取畸变系数
 * @return 畸变系数
 */
Mat CameraCalibrator::getDistCoeffs() const { return distCoeffs_; }

/**
 * @brief 获取重投影误差
 * @return 重投影误差
 */
double CameraCalibrator::getReprojectionError() const { return reprojectionError_; }
