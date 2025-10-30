#ifndef UNDISTORT_H
#define UNDISTORT_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <filesystem>

using namespace std;
using namespace cv;

namespace fs = filesystem;

class CameraCalibrator {
private:
    Size boardSize;          // 棋盘格内角点数量 (width, height)
    float squareSize;           // 棋盘格方格的实际大小 (单位: 毫米) 
    vector<vector<Point3f>> objectPoints; // 世界坐标系中的三维点
    vector<vector<Point2f>> imagePoints;  // 图像坐标系中的二维点
    Size imageSize;         // 图像尺寸

    Mat cameraMatrix;       // 相机内参矩阵
    Mat distCoeffs;         // 畸变系数
    vector<Mat> rvecs, tvecs; // 旋转和平移向量

    double reprojectionError;   // 重投影误差

public:
    CameraCalibrator(Size boardSize, float squareSize = 1.0f);

    // 准备物体点 (世界坐标系中的点) 
    vector<Point3f> prepareObjectPoints();

    // 处理单张图像，提取角点
    bool processImage(const string& imagePath, bool showResult = false);

    // 处理文件夹中的所有图像
    int processImagesFromFolder(const string& folderPath, bool showResult = false);

    // 执行相机校准
    double calibrate();

    // 校正单张图像（优化版本，避免黑边问题）
    // 参数说明：
    // - inputImage: 输入的畸变图像
    // - crop: 是否裁剪到原始图像大小
    // - borderSize: 扩展边界的大小，默认为100像素
    // - borderMode: 边界填充模式，默认为BORDER_REPLICATE
    Mat undistortImage(const Mat& inputImage, bool crop = true, int borderSize = 100, int borderMode = BORDER_REPLICATE);

    // 保存校准参数到文件
    bool saveCalibration(const string& filename);

    // 从文件加载校准参数
    bool loadCalibration(const string& filename);

    // 获取相机参数
    Mat getCameraMatrix() const;
    Mat getDistCoeffs() const;
    double getReprojectionError() const;
};

#endif // UNDISTORT_H
