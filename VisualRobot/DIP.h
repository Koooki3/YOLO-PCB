/************************************************************************/
/* 提供基于Halcon/OpenCV和Eigen的坐标变换矩阵创建、计算、保存、读取支持和现有待测物体边缘检测和拟合算法支持 */
/************************************************************************/

#ifndef DIP_H
#define DIP_H

#include <QVector>
#include <QPointF>
#include <QDir>
#include <QCoreApplication>
#include "eigen3/Eigen/Dense"
#include <halconcpp/HalconCpp.h>
#include <Halcon.h>
#include <halconcpp/HDevThread.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <vector>

using namespace HalconCpp;
using namespace Eigen;
using namespace std;

// 参数结构体，类似于原始Params
struct Params {
    int thresh = 127;
    int maxval = 255;
    int blurK = 5;
    double areaMin = 100.0;
};

// 结果结构体，用于存储输出
struct Result {
    vector<float> widths;
    vector<float> heights;
    vector<float> angles;
    cv::Mat image;
};

bool createDirectory(const string& path);

int TransMatrix(const QVector<QPointF>& WorldCoord, const QVector<QPointF>& PixelCoord, Matrix3d& matrix, const string& filename);

int Algorithm(const string& imgPath, HTuple& Row, HTuple& Col);

Matrix3d readTransformationMatrix(const string& filePath);

int getCoords(QVector<QPointF>& WorldCoord, QVector<QPointF>& PixelCoord, double size);

// OpenCV版本的函数声明
int Algorithm_opencv(const string& imgPath, vector<double>& Row, vector<double>& Col);
int getCoords_opencv(QVector<QPointF>& WorldCoord, QVector<QPointF>& PixelCoord, double size);

Result calculateLength(const cv::Mat& input, const Params& params, double bias);

#endif // DIP_H
