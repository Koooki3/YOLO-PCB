#include "Undistort.h"
#include <iostream>
#include <opencv2/opencv.hpp>

CameraCalibrator::CameraCalibrator(cv::Size boardSize, float squareSize)
    : boardSize(boardSize), squareSize(squareSize) {}

// 准备物体点 (世界坐标系中的点) 
std::vector<cv::Point3f> CameraCalibrator::prepareObjectPoints() {
    std::vector<cv::Point3f> objP;
    for (int i = 0; i < boardSize.height; i++) {
        for (int j = 0; j < boardSize.width; j++) {
            objP.push_back(cv::Point3f(j * squareSize, i * squareSize, 0.0f));
        }
    }
    return objP;
}

// 处理单张图像，提取角点
bool CameraCalibrator::processImage(const std::string& imagePath, bool showResult) {
    cv::Mat image = cv::imread(imagePath);
    if (image.empty()) {
        std::cerr << "无法读取图像: " << imagePath << std::endl;
        return false;
    }

    // 如果是第一张图像，记录图像尺寸
    if (imageSize.empty()) {
        imageSize = image.size();
    }

    cv::Mat gray;
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);

    std::vector<cv::Point2f> corners;
    bool found = cv::findChessboardCorners(gray, boardSize, corners);

    if (found) {
        // 亚像素精确化
        cv::TermCriteria criteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 30, 0.001);
        cv::cornerSubPix(gray, corners, cv::Size(11, 11), cv::Size(-1, -1), criteria);

        // 保存角点
        imagePoints.push_back(corners);
        objectPoints.push_back(prepareObjectPoints());

        if (showResult) {
            cv::drawChessboardCorners(image, boardSize, corners, found);
            cv::imshow("角点检测", image);
            cv::waitKey(0);
        }

        return true;
    } else {
        std::cerr << "未找到棋盘格角点: " << imagePath << std::endl;
        return false;
    }
}

// 处理文件夹中的所有图像
int CameraCalibrator::processImagesFromFolder(const std::string& folderPath, bool showResult) {
    int processedCount = 0;

    try {
        for (const auto& entry : fs::directory_iterator(folderPath)) {
            if (entry.is_regular_file()) {
                std::string ext = entry.path().extension().string();
                // 检查常见图像格式
                if (ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp") {
                    if (processImage(entry.path().string(), showResult)) {
                        processedCount++;
                    }
                }
            }
        }
    } catch (const fs::filesystem_error& e) {
        std::cerr << "文件系统错误: " << e.what() << std::endl;
    }

    std::cout << "成功处理 " << processedCount << " 张图像" << std::endl;
    return processedCount;
}

// 执行相机校准
double CameraCalibrator::calibrate() {
    if (imagePoints.size() < 10) {
        std::cerr << "需要至少10张有效图像进行校准，当前只有 " << imagePoints.size() << " 张" << std::endl;
        return -1;
    }

    std::vector<cv::Mat> rvecs, tvecs;
    reprojectionError = cv::calibrateCamera(
        objectPoints, imagePoints, imageSize,
        cameraMatrix, distCoeffs, rvecs, tvecs,
        cv::CALIB_FIX_K3 // 固定k3系数，通常k3影响不大且需要更多数据
    );

    std::cout << "重投影误差: " << reprojectionError << std::endl;
    std::cout << "相机内参矩阵:\n" << cameraMatrix << std::endl;
    std::cout << "畸变系数: " << distCoeffs.t() << std::endl;

    return reprojectionError;
}

// 校正单张图像
cv::Mat CameraCalibrator::undistortImage(const cv::Mat& inputImage, bool crop) {
    cv::Mat undistorted;

    // 获取优化后的新相机矩阵
    cv::Mat newCameraMatrix = cv::getOptimalNewCameraMatrix(
        cameraMatrix, distCoeffs, imageSize, 1.0, imageSize
    );

    // 校正图像
    cv::undistort(inputImage, undistorted, cameraMatrix, distCoeffs, newCameraMatrix);

    // 如果需要，裁剪黑边
    if (crop) {
        cv::Rect roi;
        cv::getOptimalNewCameraMatrix(
            cameraMatrix, distCoeffs, imageSize, 0, imageSize, &roi
        );
        undistorted = undistorted(roi);
    }

    return undistorted;
}

// 保存校准参数到文件
bool CameraCalibrator::saveCalibration(const std::string& filename) {
    cv::FileStorage fs(filename, cv::FileStorage::WRITE);
    if (!fs.isOpened()) {
        std::cerr << "无法创建文件: " << filename << std::endl;
        return false;
    }

    fs << "camera_matrix" << cameraMatrix;
    fs << "distortion_coefficients" << distCoeffs;
    fs << "reprojection_error" << reprojectionError;
    fs << "image_width" << imageSize.width;
    fs << "image_height" << imageSize.height;

    fs.release();
    std::cout << "校准参数已保存到: " << filename << std::endl;
    return true;
}

// 从文件加载校准参数
bool CameraCalibrator::loadCalibration(const std::string& filename) {
    cv::FileStorage fs(filename, cv::FileStorage::READ);
    if (!fs.isOpened()) {
        std::cerr << "无法打开文件: " << filename << std::endl;
        return false;
    }

    fs["camera_matrix"] >> cameraMatrix;
    fs["distortion_coefficients"] >> distCoeffs;
    fs["reprojection_error"] >> reprojectionError;

    int width, height;
    fs["image_width"] >> width;
    fs["image_height"] >> height;
    imageSize = cv::Size(width, height);

    fs.release();
    std::cout << "校准参数已从 " << filename << " 加载" << std::endl;
    return true;
}

// 获取相机参数
cv::Mat CameraCalibrator::getCameraMatrix() const { return cameraMatrix; }
cv::Mat CameraCalibrator::getDistCoeffs() const { return distCoeffs; }
double CameraCalibrator::getReprojectionError() const { return reprojectionError; }
