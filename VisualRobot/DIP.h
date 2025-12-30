/**
 * @file DIP.h
 * @brief 数字图像处理模块头文件
 * 
 * 该模块提供基于OpenCV和Eigen的坐标变换矩阵创建、计算、保存、读取支持，
 * 以及现有待测物体边缘检测和拟合算法支持。
 * 
 * 主要功能包括：
 * - 坐标变换矩阵的读取和保存
 * - 图像中矩形、圆形标记的检测
 * - 单目标和多目标尺寸测量
 * - YOLO检测结果处理和图像掩码
 * - 基于PCA的有向包围盒(OBB)计算
 * 
 * @author VisualRobot Team
 * @date 2025-12-29
 * @version 1.0
 */

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
 * 
 * 该结构体包含了图像预处理、轮廓检测和尺寸计算所需的各种参数。
 * 所有参数都有合理的默认值，可以根据具体应用场景进行调整。
 */
struct Params 
{
    int thresh = 127;           ///< 二值化阈值，默认127
    int maxval = 255;           ///< 二值化最大值，默认255
    int blurK = 5;              ///< 模糊核大小，默认5
    double areaMin = 8000.0;    ///< 最小轮廓面积，默认8000.0
    int mergeRadius = 8;        ///< 形态学闭运算半径，默认8
};

/**
 * @brief 结果结构体，用于存储图像处理和检测的输出结果
 * 
 * 该结构体包含了检测到的物体尺寸信息和处理后的可视化图像。
 */
struct Result 
{
    vector<float> widths;   ///< 检测到的物体宽度列表（像素或实际单位）
    vector<float> heights;  ///< 检测到的物体高度列表（像素或实际单位）
    vector<float> angles;   ///< 检测到的物体角度列表（度）
    Mat image;              ///< 处理后的图像，包含检测结果可视化
};

/**
 * @brief YOLO检测框结构体
 * 
 * 用于存储YOLO模型检测到的目标框信息，包括边界框坐标、类别和置信度。
 */
struct YoloDetBox
{
    cv::Rect2f bbox;     ///< 边界框 [xmin, ymin, xmax, ymax]
    std::string cls;     ///< 类别名称，如"defect1"、"defect2"、"module1"等
    int class_id = -1;   ///< 类别ID
    float confidence = 0.f; ///< 置信度
};

/**
 * @brief 创建目录的辅助函数
 * 
 * 使用Qt的QDir类创建指定路径的目录，包括所有必要的父目录。
 * 如果目录已存在，也会返回true。
 * 
 * @param path 要创建的目录路径
 * @return bool 创建成功返回true，失败返回false
 */
bool CreateDirectory(const string& path);

/**
 * @brief 从文件读取变换矩阵
 * 
 * 从指定文件中读取3x3变换矩阵，文件格式应为3行3列的浮点数矩阵。
 * 每行的元素之间用空格分隔。
 * 
 * @param filename 矩阵文件路径
 * @return Matrix3d 读取到的3x3变换矩阵
 * @throws runtime_error 如果文件打开失败或解析错误
 */
Matrix3d ReadTransformationMatrix(const string& filename);

/**
 * @brief 使用OpenCV检测图像中的矩形
 * 
 * 该函数实现了基于OpenCV的矩形检测算法，主要步骤包括：
 * 1. 读取图像
 * 2. 图像预处理（灰度化、二值化、平滑处理）
 * 3. 使用Canny边缘检测
 * 4. 轮廓提取和过滤
 * 5. 寻找最大的矩形轮廓
 * 6. 亚像素精度优化角点坐标
 * 7. 可视化结果并保存
 * 
 * @param imgPath 输入图像路径
 * @param Row 输出参数，存储矩形角点的行坐标
 * @param Col 输出参数，存储矩形角点的列坐标
 * @return int 成功返回0，失败返回1
 * @note 检测到的矩形角点按照顺时针顺序存储，顺序为：左上角、右上角、左下角、右下角
 */
int DetectRectangleOpenCV(const string& imgPath, vector<double>& Row, vector<double>& Col);

/**
 * @brief 使用OpenCV获取标定板坐标
 * 
 * 该函数实现了基于OpenCV的圆形检测算法，用于检测标定板上的圆形标记。
 * 主要步骤包括：
 * 1. 读取图像
 * 2. 计算圆形检测参数
 * 3. 定义搜索区域
 * 4. 图像预处理（灰度化、二值化）
 * 5. 使用Hough圆变换检测圆
 * 6. 亚像素精度优化圆心坐标
 * 7. 存储结果并绘制
 * 
 * @param WorldCoord 输出参数，存储检测到的世界坐标
 * @param PixelCoord 输出参数，存储检测到的像素坐标
 * @param size 标定板尺寸，默认为100.0
 * @return int 成功返回0，失败返回1
 */
int GetCoordsOpenCV(QVector<QPointF>& WorldCoord, QVector<QPointF>& PixelCoord, double size = 100.0);

/**
 * @brief 处理图像并计算单个物体的尺寸
 * 
 * 该函数实现了单个物体的尺寸测量功能，主要步骤包括：
 * 1. 使用YOLO检测结果进行图像掩码（可选）
 * 2. 图像预处理（灰度化、滤波、二值化、形态学操作）
 * 3. 轮廓提取和过滤
 * 4. 寻找最大面积的轮廓
 * 5. 使用PCA计算有向包围盒(OBB)
 * 6. 计算物体尺寸和角度
 * 7. 绘制结果并返回
 * 
 * @param input 输入图像
 * @param params 处理参数，包含阈值、模糊核大小等
 * @param bias 比例偏差，用于将像素单位转换为实际单位
 * @return Result 包含宽度、高度、角度和处理后图像的结果结构体
 * @see CalculateLengthMultiTarget()
 */
Result CalculateLength(const cv::Mat& input, const Params& params, double bias);

/**
 * @brief 基于连通域的多目标尺寸测量函数
 * 
 * 该函数实现了基于连通域的多目标检测与测量算法，主要步骤包括：
 * 1. 图像预处理（灰度化、滤波、二值化、形态学操作）
 * 2. 轮廓提取和面积过滤
 * 3. 第一阶段：处理所有轮廓，收集信息并执行Canny边缘检测
 * 4. 边缘处理：排除与轮廓近似的边缘，对剩余边缘进行平滑拟合
 * 5. 第二阶段：绘制轮廓边界框和序号，根据边缘情况决定颜色
 * 6. 计算每个目标的尺寸和角度，并保存结果
 * 
 * @param input 输入图像
 * @param params 处理参数，包含阈值、模糊核大小、面积最小值等
 * @param bias 比例偏差，用于将像素单位转换为实际物理单位
 * @return Result 包含所有检测目标的宽度、高度、角度和结果图像的结构体
 * @note 检测到的目标将按照从1开始的序号进行标记，边界框颜色根据是否存在非轮廓边缘决定：
 *       - 红色：存在非轮廓边缘
 *       - 绿色：无非轮廓边缘
 * @see CalculateLength()
 */
Result CalculateLengthMultiTarget(const Mat& input, const Params& params, double bias);

/**
 * @brief 使用PCA主轴方向计算点集的有向包围盒(OBB)
 * 
 * 该函数使用PCA（主成分分析）计算点集的主轴方向，并返回与主轴对齐的旋转矩形。
 * 用于替代minAreaRect以获得更符合"主轴/最长方向"的旋转框。
 * 
 * @param pts 输入点集（通常是某个连通域/轮廓的像素点）
 * @return RotatedRect PCA主轴对齐的旋转矩形
 * @note 如果点数少于10个，会退化为minAreaRect以避免错误
 */
RotatedRect GetOBBByPCA(const vector<Point>& pts);

/**
 * @brief 从JSON文件读取YOLO检测结果
 * 
 * 读取包含YOLO检测结果的JSON文件，解析为YoloDetBox结构体列表。
 * 只保留类别为"defect1"或"defect2"的检测结果。
 * 
 * @param jsonPath JSON文件路径
 * @param dets 输出参数，存储解析后的检测框列表
 * @return bool 读取成功返回true，失败返回false
 */
bool LoadYoloDetections(const std::string& jsonPath, std::vector<YoloDetBox>& dets);

/**
 * @brief 放大边界框
 * 
 * 根据指定的比例放大边界框，并确保结果在图像范围内。
 * 
 * @param box 输入边界框
 * @param imgW 图像宽度
 * @param imgH 图像高度
 * @param scale 放大比例
 * @return cv::Rect 放大后的边界框
 */
cv::Rect EnlargeBBox(const cv::Rect2f& box, int imgW, int imgH, float scale);

/**
 * @brief 根据YOLO检测结果对图像进行掩码处理
 * 
 * 读取YOLO检测结果，将检测到的缺陷区域放大后涂黑，只保留非缺陷区域。
 * 
 * @param input 输入图像
 * @param jsonPath YOLO检测结果JSON文件路径
 * @param scale 边界框放大比例
 * @return cv::Mat 掩码处理后的图像
 */
cv::Mat MaskImageByYoloJson(const cv::Mat& input, const std::string& jsonPath, float scale);

#endif // DIP_H
