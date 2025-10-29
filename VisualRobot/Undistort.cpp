#include "Undistort.h"
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

CameraCalibrator::CameraCalibrator(Size boardSize, float squareSize)
    : boardSize(boardSize), squareSize(squareSize) {}

// 准备物体点 (世界坐标系中的点) 
vector<Point3f> CameraCalibrator::prepareObjectPoints()
{
    vector<Point3f> objP;
    for (int i = 0; i < boardSize.height; i++)
    {
        for (int j = 0; j < boardSize.width; j++)
        {
            objP.push_back(Point3f(j * squareSize, i * squareSize, 0.0f));
        }
    }
    return objP;
}

// 处理单张图像，提取角点
bool CameraCalibrator::processImage(const string& imagePath, bool showResult)
{
    Mat image = imread(imagePath);
    vector<string> imagePaths;
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

    Mat gray;
    cvtColor(image, gray, COLOR_BGR2GRAY);

    vector<Point2f> corners;
    bool found = findChessboardCorners(gray, boardSize, corners);

    if (found)
    {
        // 亚像素精确化
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

            // 绘制像素坐标（白色）
            putText(annotatedImage, pixelText, textPos, FONT_HERSHEY_SIMPLEX, 0.4, Scalar(0, 255, 0), 1);

            // 绘制世界坐标（黄色）
            putText(annotatedImage, worldText, Point(textPos.x, textPos.y + 15), FONT_HERSHEY_SIMPLEX, 0.4, Scalar(0, 255, 255), 1);
        }

        // 保存处理后的图像
        fs::path pathObj(imagePath);
        string processedPath = pathObj.parent_path().string() + "/" + pathObj.stem().string() + "_processed" + pathObj.extension().string();
        imwrite(processedPath, annotatedImage);
        cout << "已保存处理结果: " << processedPath << endl;

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

// 处理文件夹中的所有图像
int CameraCalibrator::processImagesFromFolder(const string& folderPath, bool showResult)
{
    int processedCount = 0;

    try
    {
        for (const auto& entry : fs::directory_iterator(folderPath))
        {
            if (entry.is_regular_file())
            {
                string ext = entry.path().extension().string();
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
        cerr << "文件系统错误: " << e.what() << endl;
    }

    cout << "成功处理 " << processedCount << " 张图像" << endl;
    return processedCount;
}

// 执行相机校准
double CameraCalibrator::calibrate()
{
    if (imagePoints.size() < 10)
    {
        cerr << "需要至少10张有效图像进行校准，当前只有 " << imagePoints.size() << " 张" << endl;
        return -1;
    }

    vector<Mat> rvecs, tvecs;
    reprojectionError = calibrateCamera(
        objectPoints, imagePoints, imageSize,
        cameraMatrix, distCoeffs, rvecs, tvecs,
        CALIB_FIX_K3 // 固定k3系数，通常k3影响不大且需要更多数据
    );

    cout << "重投影误差: " << reprojectionError << endl;
    cout << "相机内参矩阵:\n" << cameraMatrix << endl;
    cout << "畸变系数: " << distCoeffs.t() << endl;

    return reprojectionError;
}

// 校正单张图像
Mat CameraCalibrator::undistortImage(const Mat& inputImage, bool crop)
{
    Mat undistorted;

    // 获取优化后的新相机矩阵
    Mat newCameraMatrix = getOptimalNewCameraMatrix(cameraMatrix, distCoeffs, imageSize, 1.0, imageSize);

    // 校正图像
    undistort(inputImage, undistorted, cameraMatrix, distCoeffs, newCameraMatrix);

    // 如果需要，裁剪黑边
    if (crop)
    {
        Rect roi;
        getOptimalNewCameraMatrix(cameraMatrix, distCoeffs, imageSize, 0, imageSize, &roi);
        undistorted = undistorted(roi);
    }

    return undistorted;
}

// 保存校准参数到文件
bool CameraCalibrator::saveCalibration(const string& filename)
{
    FileStorage fs(filename, FileStorage::WRITE);
    if (!fs.isOpened())
    {
        cerr << "无法创建文件: " << filename << endl;
        return false;
    }

    fs << "camera_matrix" << cameraMatrix;
    fs << "distortion_coefficients" << distCoeffs;
    fs << "reprojection_error" << reprojectionError;
    fs << "image_width" << imageSize.width;
    fs << "image_height" << imageSize.height;

    fs.release();
    cout << "校准参数已保存到: " << filename << endl;
    return true;
}

// 从文件加载校准参数
bool CameraCalibrator::loadCalibration(const string& filename)
{
    FileStorage fs(filename, FileStorage::READ);
    if (!fs.isOpened())
    {
        cerr << "无法打开文件: " << filename << endl;
        return false;
    }

    fs["camera_matrix"] >> cameraMatrix;
    fs["distortion_coefficients"] >> distCoeffs;
    fs["reprojection_error"] >> reprojectionError;

    int width, height;
    fs["image_width"] >> width;
    fs["image_height"] >> height;
    imageSize = Size(width, height);

    fs.release();
    cout << "校准参数已从 " << filename << " 加载" << endl;
    return true;
}

// 获取相机参数
Mat CameraCalibrator::getCameraMatrix() const { return cameraMatrix; }
Mat CameraCalibrator::getDistCoeffs() const { return distCoeffs; }
double CameraCalibrator::getReprojectionError() const { return reprojectionError; }
