#include "Undistort.h"
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

/**
 * @brief CameraCalibrator构造函数
 * @param boardSize 棋盘格内角点数量 (width, height)
 * @param squareSize 棋盘格方格的实际大小 (单位: 毫米)
 */
CameraCalibrator::CameraCalibrator(Size boardSize, float squareSize)
    : boardSize(boardSize), squareSize(squareSize) {}

/**
 * @brief 准备物体点 (世界坐标系中的点)
 * @return 世界坐标系中的三维点集合
 * 
 * 生成棋盘格在世界坐标系中的三维点，用于相机校准
 */
vector<Point3f> CameraCalibrator::prepareObjectPoints()
{
    vector<Point3f> objP;
    for (int i = 0; i < boardSize.height; i++)
    {
        for (int j = 0; j < boardSize.width; j++)
        {
            // 生成世界坐标系中的三维点，z坐标为0
            objP.push_back(Point3f(j * squareSize, i * squareSize, 0.0f));
        }
    }
    return objP;
}

/**
 * @brief 处理单张图像，提取角点
 * @param imagePath 图像文件路径
 * @param showResult 是否显示提取结果
 * @return 角点提取是否成功
 * 
 * 从单张图像中提取棋盘格角点，并保存到图像点集合中
 */
bool CameraCalibrator::processImage(const string& imagePath, bool showResult)
{
    Mat image = imread(imagePath);
    vector<string> imagePaths;
    
    // 检查图像是否读取成功
    if (image.empty())
    {
        cerr << "无法读取图像: " << imagePath << endl;
        return false;
    }

    // 如果是第一张图像，记录图像尺寸
    if (imageSize.empty())
    {
        imageSize = image.size();
    }

    // 转换为灰度图像
    Mat gray;
    cvtColor(image, gray, COLOR_BGR2GRAY);

    // 查找棋盘格角点
    vector<Point2f> corners;
    bool found = findChessboardCorners(gray, boardSize, corners);

    if (found)
    {
        // 亚像素精确化，提高角点检测精度
        TermCriteria criteria(TermCriteria::EPS + TermCriteria::MAX_ITER, 30, 0.001);
        cornerSubPix(gray, corners, Size(11, 11), Size(-1, -1), criteria);

        // 保存角点
        imagePoints.push_back(corners);
        imagePaths.push_back(imagePath); // 保存图像路径
        vector<Point3f> objPoints = prepareObjectPoints();
        objectPoints.push_back(objPoints);

        // 创建带标注的图像
        Mat annotatedImage = image.clone();
        drawChessboardCorners(annotatedImage, boardSize, corners, found);

        // 在图像上标注每个角点的坐标
        for (size_t i = 0; i < corners.size(); i++)
        {
            Point2f pixelCoord = corners[i];
            Point3f worldCoord = objPoints[i];

            // 创建坐标文本
            string pixelText = "Pix: (" + to_string((double)pixelCoord.x) + ", " + to_string((double)pixelCoord.y) + ")";
            string worldText = "World: (" + to_string(worldCoord.x) + ", " + to_string(worldCoord.y) + ", " + to_string(worldCoord.z) + ")";

            // 设置文本位置（稍微偏移以避免重叠）
            Point textPos(pixelCoord.x + 10, pixelCoord.y - 10);

            // 绘制像素坐标（绿色）
            putText(annotatedImage, pixelText, textPos, FONT_HERSHEY_SIMPLEX, 0.4, Scalar(0, 255, 0), 1);

            // 绘制世界坐标（黄色）
            putText(annotatedImage, worldText, Point(textPos.x, textPos.y + 15), FONT_HERSHEY_SIMPLEX, 0.4, Scalar(0, 255, 255), 1);
        }

        // 保存处理后的图像
        fs::path pathObj(imagePath);
        string processedPath = pathObj.parent_path().string() + "/" + pathObj.stem().string() + "_processed" + pathObj.extension().string();
        imwrite(processedPath, annotatedImage);
        cout << "已保存处理结果: " << processedPath << endl;

        // 如果需要显示结果
        if (showResult)
        {
            imshow("角点检测", annotatedImage);
            waitKey(0);
        }

        return true;
    }
    else
    {
        cerr << "未找到棋盘格角点: " << imagePath << endl;
        return false;
    }
}

/**
 * @brief 处理文件夹中的所有图像
 * @param folderPath 文件夹路径
 * @param showResult 是否显示提取结果
 * @return 成功处理的图像数量
 * 
 * 从文件夹中遍历所有图像，提取棋盘格角点，并保存到图像点集合中
 */
int CameraCalibrator::processImagesFromFolder(const string& folderPath, bool showResult)
{
    int processedCount = 0;

    try
    {
        // 遍历文件夹中的所有文件
        for (const auto& entry : fs::directory_iterator(folderPath))
        {
            if (entry.is_regular_file())
            {
                string ext = entry.path().extension().string();
                // 检查常见图像格式
                if (ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp")
                {
                    // 处理图像
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
        cerr << "文件系统错误: " << e.what() << endl;
    }

    cout << "成功处理 " << processedCount << " 张图像" << endl;
    return processedCount;
}

/**
 * @brief 执行相机校准
 * @return 重投影误差
 * 
 * 使用提取的角点执行相机校准，计算相机内参和畸变系数
 */
double CameraCalibrator::calibrate()
{
    // 检查是否有足够的图像进行校准
    if (imagePoints.size() < 3)
    {
        cerr << "需要至少3张有效图像进行校准，当前只有 " << imagePoints.size() << " 张" << endl;
        return -1;
    }

    vector<Mat> rvecs, tvecs;
    
    // 执行相机校准
    reprojectionError = calibrateCamera(
        objectPoints, imagePoints, imageSize,
        cameraMatrix, distCoeffs, rvecs, tvecs,
        CALIB_FIX_K3 // 固定k3系数，通常k3影响不大且需要更多数据
    );

    // 输出校准结果
    cout << "重投影误差: " << reprojectionError << endl;
    cout << "相机内参矩阵:\n" << cameraMatrix << endl;
    cout << "畸变系数: " << distCoeffs.t() << endl;

    return reprojectionError;
}

/**
 * @brief 校正单张图像（优化版本，避免黑边问题）
 * @param inputImage 输入的畸变图像
 * @param crop 是否裁剪到原始图像大小
 * @param borderSize 扩展边界的大小，默认为100像素
 * @param borderMode 边界填充模式，默认为BORDER_REPLICATE
 * @return 校正后的图像
 * 
 * 对输入的畸变图像进行校正，支持多种参数配置，避免校正后图像出现黑边问题
 */
Mat CameraCalibrator::undistortImage(const Mat& inputImage, bool crop, int borderSize, int borderMode)
{
    Mat undistorted;
    
    // 检查输入图像是否为空
    if (inputImage.empty()) 
    {
        cerr << "输入图像为空" << endl;
        return undistorted;
    }
    
    // 方案：使用扩展边界+remap方法避免黑边问题
    // 1. 扩展输入图像边界，确保所有目标像素都能找到对应的源像素
    Mat extendedImage;
    copyMakeBorder(inputImage, extendedImage, borderSize, borderSize, borderSize, borderSize, borderMode);
    
    // 2. 为扩展后的图像准备新的相机矩阵和尺寸
    Size extendedSize = extendedImage.size();
    Mat extendedCameraMatrix = cameraMatrix.clone();
    // 调整相机矩阵的光心坐标，考虑扩展的边界
    extendedCameraMatrix.at<double>(0, 2) += borderSize;
    extendedCameraMatrix.at<double>(1, 2) += borderSize;
    
    // 3. 获取优化后的新相机矩阵（alpha=1.0保留完整图像）
    Mat newCameraMatrix = getOptimalNewCameraMatrix(extendedCameraMatrix, distCoeffs, extendedSize, 1.0, extendedSize);
    
    // 4. 生成映射数组
    Mat mapX, mapY;
    initUndistortRectifyMap(
        extendedCameraMatrix, distCoeffs, Mat(), 
        newCameraMatrix, extendedSize, CV_32FC1, mapX, mapY
    );
    
    // 5. 使用remap进行去畸变，指定边界模式
    Mat remappedImage;
    remap(extendedImage, remappedImage, mapX, mapY, INTER_LINEAR, borderMode);
    
    // 6. 处理裁剪选项
    if (crop) 
    {
        // 计算裁剪区域（不包括扩展的边界）
        Rect roi(borderSize, borderSize, inputImage.cols, inputImage.rows);
        undistorted = remappedImage(roi);
    } 
    else 
    {
        // 返回完整的去畸变图像
        undistorted = remappedImage;
    }
    
    return undistorted;
}

/**
 * @brief 保存校准参数到文件
 * @param filename 保存文件名
 * @return 保存是否成功
 * 
 * 将相机内参、畸变系数等校准参数保存到文件中
 */
bool CameraCalibrator::saveCalibration(const string& filename)
{
    // 打开文件存储
    FileStorage fs(filename, FileStorage::WRITE);
    if (!fs.isOpened())
    {
        cerr << "无法创建文件: " << filename << endl;
        return false;
    }

    // 保存校准参数
    fs << "camera_matrix" << cameraMatrix;
    fs << "distortion_coefficients" << distCoeffs;
    fs << "reprojection_error" << reprojectionError;
    fs << "image_width" << imageSize.width;
    fs << "image_height" << imageSize.height;

    // 释放文件存储
    fs.release();
    cout << "校准参数已保存到: " << filename << endl;
    return true;
}

/**
 * @brief 从文件加载校准参数
 * @param filename 加载文件名
 * @return 加载是否成功
 * 
 * 从文件中加载相机内参、畸变系数等校准参数
 */
bool CameraCalibrator::loadCalibration(const string& filename)
{
    // 打开文件存储
    FileStorage fs(filename, FileStorage::READ);
    if (!fs.isOpened())
    {
        cerr << "无法打开文件: " << filename << endl;
        return false;
    }

    // 加载校准参数
    fs["camera_matrix"] >> cameraMatrix;
    fs["distortion_coefficients"] >> distCoeffs;
    fs["reprojection_error"] >> reprojectionError;

    // 加载图像尺寸
    int width, height;
    fs["image_width"] >> width;
    fs["image_height"] >> height;
    imageSize = Size(width, height);

    // 释放文件存储
    fs.release();
    cout << "校准参数已从 " << filename << " 加载" << endl;
    return true;
}

/**
 * @brief 获取相机内参矩阵
 * @return 相机内参矩阵
 */
Mat CameraCalibrator::getCameraMatrix() const { return cameraMatrix; }

/**
 * @brief 获取畸变系数
 * @return 畸变系数
 */
Mat CameraCalibrator::getDistCoeffs() const { return distCoeffs; }

/**
 * @brief 获取重投影误差
 * @return 重投影误差
 */
double CameraCalibrator::getReprojectionError() const { return reprojectionError; }
