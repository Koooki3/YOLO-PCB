/************************************************************************/
/* 提供基于OpenCV和Eigen的坐标变换矩阵创建、计算、保存、读取支持和现有待测物体边缘检测和拟合算法支持 */
/************************************************************************/

#ifndef DIP_H
#define DIP_H

#include <QVector>
#include <QPointF>
#include <QDir>
#include <QCoreApplication>
#include "eigen3/Eigen/Dense"
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <vector>
#include <fstream>
#include <nlohmann/json.hpp>

using namespace Eigen;
using namespace std;
using namespace cv;
using json = nlohmann::json;

/**
 * @brief 参数结构体，用于配置图像处理和检测算法的参数
 */
struct Params 
{
    int thresh = 127;     ///< 二值化阈值，默认127
    int maxval = 255;     ///< 二值化最大值，默认255
    int blurK = 5;        ///< 模糊核大小，默认5
    double areaMin = 8000.0; ///< 最小轮廓面积，默认5000.0
    int mergeRadius = 8;    // 你新增的参数
};

/**
 * @brief 结果结构体，用于存储图像处理和检测的输出结果
 */
struct Result 
{
    vector<float> widths;   ///< 检测到的物体宽度列表
    vector<float> heights;  ///< 检测到的物体高度列表
    vector<float> angles;   ///< 检测到的物体角度列表
    Mat image;              ///< 处理后的图像，包含检测结果可视化
};

// YOLO 检测框结构
struct YoloDetBox
{
    cv::Rect2f bbox;     // [xmin, ymin, xmax, ymax] -> Rect2f
    std::string cls;     // "defect1" / "defect2" / "module1" ...
    int class_id = -1;
    float confidence = 0.f;
};

/**
 * @brief 创建目录的辅助函数
 * @param path 要创建的目录路径
 * @return 创建成功返回true，失败返回false
 */
bool CreateDirectory(const string& path);

/**
 * @brief 从文件读取变换矩阵
 * @param filename 矩阵文件路径
 * @return 读取到的3x3变换矩阵
 * @throws runtime_error 如果文件打开失败或解析错误
 */
Matrix3d ReadTransformationMatrix(const string& filename);

/**
 * @brief 使用OpenCV检测图像中的矩形
 * @param imgPath 输入图像路径
 * @param Row 输出参数，存储矩形角点的行坐标
 * @param Col 输出参数，存储矩形角点的列坐标
 * @return 成功返回0，失败返回1
 */
int DetectRectangleOpenCV(const string& imgPath, vector<double>& Row, vector<double>& Col);

/**
 * @brief 使用OpenCV获取标定板坐标
 * @param WorldCoord 输出参数，存储检测到的世界坐标
 * @param PixelCoord 输出参数，存储检测到的像素坐标
 * @param size 标定板尺寸，默认100.0
 * @return 成功返回0，失败返回1
 */
int GetCoordsOpenCV(QVector<QPointF>& WorldCoord, QVector<QPointF>& PixelCoord, double size);

/**
 * @brief 处理图像并计算单个物体的尺寸
 * @param input 输入图像
 * @param params 处理参数
 * @param bias 比例偏差，用于将像素单位转换为实际单位
 * @return 包含宽度、高度、角度和处理后图像的结果结构体
 */
Result CalculateLength(const Mat& input, const Params& params, double bias);

/**
 * @brief 基于连通域的多目标尺寸测量函数
 * @param input 输入图像
 * @param params 处理参数
 * @param bias 比例偏差，用于将像素单位转换为实际单位
 * @return 包含所有检测到物体的宽度、高度、角度和处理后图像的结果结构体
 */
Result CalculateLengthMultiTarget(const Mat& input, const Params& params, double bias);

/**
 * @brief 使用PCA主轴方向计算点集的有向包围盒(OBB)，用于替代minAreaRect以获得更符合“主轴/最长方向”的旋转框
 * @param pts 输入点集（通常是某个连通域/轮廓的像素点）
 * @return PCA主轴对齐的旋转矩形
 */
RotatedRect GetOBBByPCA(const vector<Point>& pts);

#endif // DIP_H
