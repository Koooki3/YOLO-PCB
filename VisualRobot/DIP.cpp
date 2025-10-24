#include "DIP.h"
#include <QVector>
#include <QPointF>
#include <QDir>
#include <QCoreApplication>
#include "eigen3/Eigen/Dense"
#include <fstream>
#include <stdexcept>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <string>
#include <cstring>
#include <QFile>
#include <QDataStream>
#include <QString>
#include <QDebug>
#include <cmath>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>`
#include <vector>
#include <algorithm>

#define ERROR 2
#define WARNNING 1
#define INFO 0

using namespace Eigen;
using namespace std;
using namespace cv;

// 创建目录的辅助函数
bool CreateDirectory(const string& path)
{
    // 变量定义
    QDir dir(QString::fromStdString(path)); // QDir对象用于目录操作
    
    return dir.mkpath("."); // 创建路径及其所有父目录
}

// 计算变换矩阵并保存到文件
// 参数:
//   WorldCoord - 世界坐标(机械坐标)向量
//   PixelCoord - 像素坐标向量
//   matrix - 输出参数，用于存储计算得到的变换矩阵
//   filename - 保存矩阵的文件名
// 返回值:
//   成功返回0，失败返回非零错误码

//int CalculateTransformationMatrix(const QVector<QPointF>& WorldCoord, const QVector<QPointF>& PixelCoord, Matrix3d& matrix, const string& filename = "")
//{
//    // 变量定义
//    const int pointCount = WorldCoord.count(); // 坐标点数量
//    int row1, row2;                            // 矩阵行索引
//    double u, v, x, y;                         // 临时坐标变量
//    Vector2d pixel_mean = Vector2d::Zero();    // 像素坐标均值
//    Vector2d world_mean = Vector2d::Zero();    // 世界坐标均值
//    double pixel_scale = 0.0;                  // 像素坐标尺度
//    double world_scale = 0.0;                  // 世界坐标尺度
//    Vector2d pixel_center;                     // 像素中心点
//    double distance, weight;                   // 距离和权重变量
//    VectorXd weights;                          // 权重向量
//    MatrixXd W;                                // 权重矩阵
//    MatrixXd ATWA;                             // A^T * W * A矩阵
//    VectorXd ATWb;                             // A^T * W * b向量
//    VectorXd x;                                // 解向量
//    VectorXd residual;                         // 残差向量
//    double current_error, prev_error;          // 误差变量
//    VectorXd correction;                       // 修正向量
//    VectorXd final_residual;                   // 最终残差
//    double final_error;                        // 最终误差
//    Vector3d pixel, world;                     // 坐标向量
//    double error_x, error_y;                   // 坐标误差
//    string filePath;                           // 文件路径
    
//    // 参数校验
//    if (WorldCoord.count() != PixelCoord.count())
//    {
//        cerr << "错误: 世界坐标与像素坐标数量不匹配" << endl;
//        return 1; // 错误码1: 坐标数量不匹配
//    }

//    if (pointCount < 3)
//    {
//        cerr << "错误: 至少需要3个对应点" << endl;
//        return 2; // 错误码2: 点数不足
//    }

//    try
//    {
//        // 预分配内存并避免重复计算
//        MatrixXd A = MatrixXd::Zero(2 * pointCount, 6); // 系数矩阵
//        VectorXd b = VectorXd::Zero(2 * pointCount); // 常数向量

//        // 使用更高效的矩阵填充方式
//        for (int i = 0; i < pointCount; ++i)
//        {
//            u = PixelCoord[i].x();
//            v = PixelCoord[i].y();
//            x = WorldCoord[i].x();
//            y = WorldCoord[i].y();

//            row1 = 2 * i;
//            row2 = 2 * i + 1;

//            // x坐标方程: x = a*u + b*v + c
//            A(row1, 0) = u;
//            A(row1, 1) = v;
//            A(row1, 2) = 1.0;
//            b(row1) = x;

//            // y坐标方程: y = d*u + e*v + f
//            A(row2, 3) = u;
//            A(row2, 4) = v;
//            A(row2, 5) = 1.0;
//            b(row2) = y;
//        }

//        // 精度优化1: 数据归一化处理 (提高数值稳定性) 
//        pixel_mean = Vector2d::Zero();
//        world_mean = Vector2d::Zero();

//        for (int i = 0; i < pointCount; ++i)
//        {
//            pixel_mean += Vector2d(PixelCoord[i].x(), PixelCoord[i].y());
//            world_mean += Vector2d(WorldCoord[i].x(), WorldCoord[i].y());
//        }
//        pixel_mean /= pointCount;
//        world_mean /= pointCount;

//        pixel_scale = 0.0;
//        world_scale = 0.0;
//        for (int i = 0; i < pointCount; ++i)
//        {
//            pixel_scale += (Vector2d(PixelCoord[i].x(), PixelCoord[i].y()) - pixel_mean).norm();
//            world_scale += (Vector2d(WorldCoord[i].x(), WorldCoord[i].y()) - world_mean).norm();
//        }
//        pixel_scale /= pointCount;
//        world_scale /= pointCount;

//        // 高级精度优化1: 加权最小二乘法
//        // 根据点的质量分配权重 (距离中心越近的点权重越高) 
//        weights = VectorXd::Ones(2 * pointCount);
//        pixel_center = pixel_mean;
//        for (int i = 0; i < pointCount; ++i)
//        {
//            Vector2d pixel_vec(PixelCoord[i].x(), PixelCoord[i].y());
//            distance = (pixel_vec - pixel_center).norm();
//            weight = exp(-distance * distance / (2.0 * pixel_scale * pixel_scale));
//            weights(2*i) = weight;
//            weights(2*i+1) = weight;
//        }

//        // 构建加权矩阵
//        W = weights.asDiagonal();
//        ATWA = A.transpose() * W * A;
//        ATWb = A.transpose() * W * b;

//        // 高级精度优化2: 使用更稳定的求解方法
//        x = ATWA.completeOrthogonalDecomposition().solve(ATWb);

//        // 高级精度优化3: 多次迭代精化
//        const int max_refinement_iterations = 3;
//        prev_error = numeric_limits<double>::max();

//        for (int iter = 0; iter < max_refinement_iterations; ++iter)
//        {
//            residual = b - A * x;
//            current_error = residual.norm();

//            // 如果误差不再显著减小，停止迭代
//            if (abs(prev_error - current_error) < 1e-12 * prev_error)
//            {
//                break;
//            }
//            prev_error = current_error;

//            // 求解修正量
//            correction = ATWA.completeOrthogonalDecomposition().solve(A.transpose() * W * residual);
//            x += correction;
//        }

//        // 高级精度优化4: RANSAC异常点检测 (可选) 
//        // 可以添加RANSAC来检测和移除异常点，进一步提高精度

//        // 计算最终误差并输出
//        final_residual = b - A * x;
//        final_error = final_residual.norm();
//        cout << "最终残差范数: " << scientific << setprecision(6) << final_error << endl;

//        // 输出每个点的误差
//        cout << "各点误差分析:" << endl;
//        for (int i = 0; i < pointCount; ++i)
//        {
//            Vector2d pixel_err(final_residual(2*i), final_residual(2*i+1));
//            cout << "点 " << i << ": 误差 = " << pixel_err.norm() << " pixels" << endl;
//        }

//        // 构建变换矩阵
//        matrix << x(0), x(1), x(2),
//                  x(3), x(4), x(5),
//                  0.0,  0.0,  1.0;

//        // 输出矩阵信息
//        cout << "变换矩阵:" << endl;
//        cout << fixed << setprecision(10);
//        cout << matrix << endl;

//        // 验证变换矩阵
//        cout << "验证变换结果:" << endl;
//        for (int i = 0; i < pointCount; ++i)
//        {
//            pixel = Vector3d(PixelCoord[i].x(), PixelCoord[i].y(), 1.0);
//            world = matrix * pixel;

//            error_x = abs(world[0] - WorldCoord[i].x());
//            error_y = abs(world[1] - WorldCoord[i].y());

//            cout << "像素坐标: (" << pixel[0] << ", " << pixel[1] << ") -> ";
//            cout << "计算的世界坐标: (" << world[0] << ", " << world[1] << ") ";
//            cout << "实际的世界坐标: (" << WorldCoord[i].x() << ", " << WorldCoord[i].y() << ")" << endl;
//            cout << " 误差: (" << error_x << ", " << error_y << ")" << endl;
//        }

//        // 如果提供了文件名，保存矩阵到文件
//        if (!filename.empty())
//        {
//            // 使用相对路径 "../matrix.bin"
//            filePath = "../matrix.bin";

//            // 保存矩阵到文件
//            ofstream fout(filePath, ios::binary);
//            if (fout.is_open())
//            {
//                fout << fixed << setprecision(10) << matrix << endl;
//                fout.close();
//                cout << "矩阵保存成功: " << filePath << endl;
//            }
//            else
//            {
//                cerr << "错误: 无法打开文件 " << filePath << endl;
//                return 4; // 错误码4: 文件打开失败
//            }
//        }

//        return 0; // 成功

//    }
//    catch (const exception& e)
//    {
//        cerr << "错误: " << e.what() << endl;
//        return 6; // 错误码6: 计算过程中发生异常
//    }
//}


// OpenCV版本的圆形检测算法
// 参数:
//   WorldCoord - 输出参数，存储检测到的世界坐标
//   PixelCoord - 输出参数，存储检测到的像素坐标  
//   size - 标定板尺寸，默认为100.0
// 返回值:
//   成功返回0，失败返回1
int GetCoordsOpenCV(QVector<QPointF>& WorldCoord, QVector<QPointF>& PixelCoord, double size = 100.0)
{
    // 变量定义
    Mat image;                              // 输入图像
    int height, width; // 图像高度和宽度
    double r, dis_col, dis_row, rmin, rmax; // 圆形检测参数
    vector<Rect> searchAreas;               // 搜索区域列表
    vector<Point2f> worldCoords;            // 世界坐标列表
    double world_dis_col, world_dis_row;    // 世界坐标间距
    Mat grayImage, binaryImage;             // 灰度图和二值图
    vector<Vec3f> detectedCircles;          // 检测到的圆形
    Mat result;                             // 结果图像
    vector<Point2f> detectedCenters;        // 检测到的圆心
    vector<float> detectedRadii;            // 检测到的半径
    vector<Point2f> detectedWorldCoords;    // 检测到的世界坐标
    int radius, roi_size, x, y;             // ROI相关变量
    Moments m;                              // 图像矩
    double cx, cy;                          // 质心坐标
    
    // 读取图像
    image = imread("/home/orangepi/Desktop/VisualRobot_Local/Img/capture.jpg");
    if(image.empty()) 
    {
        qDebug() << "Error: Cannot read image file";
        return 1;
    }

    // 获取图像尺寸
    height = image.rows;
    width = image.cols;

    // 计算参数 (基于原始代码中的比例) 
    r = (height * 316.0) / 2182.0;
    dis_col = (width * 1049.0) / 2734.0;
    dis_row = (height * 774.0) / 2182.0;
    rmin = r - 10;
    rmax = r + 10;
    
    // 计算世界坐标
    world_dis_col = 1049.0 * size / 2734.0;
    world_dis_row = 774.0 * size / 2734.0;

    // 定义9个区域的世界坐标
    for(int row = 0; row < 3; row++) 
    {
        for(int col = 0; col < 3; col++) 
        {
            // 计算搜索区域
            int x1 = (col == 0) ? 0 : static_cast<int>(r + (col - 0.5) * dis_col);
            int x2 = (col == 2) ? width : static_cast<int>(r + (col + 0.5) * dis_col);
            int y1 = (row == 0) ? 0 : static_cast<int>(r + (row - 0.5) * dis_row);
            int y2 = (row == 2) ? height : static_cast<int>(r + (row + 0.5) * dis_row);
            
            searchAreas.push_back(Rect(x1, y1, x2-x1, y2-y1));
            worldCoords.push_back(Point2f(col * world_dis_col, row * world_dis_row));
        }
    }

    // 转换为灰度图
    cvtColor(image, grayImage, COLOR_BGR2GRAY);

    // 二值化
    threshold(grayImage, binaryImage, 200, 255, THRESH_BINARY);

    // 在每个区域中检测圆
    result = image.clone();

    for(size_t i = 0; i < searchAreas.size(); i++) 
    {
        Mat roi = binaryImage(searchAreas[i]);
        vector<Vec3f> circles;

        // 使用Hough圆变换检测圆
        HoughCircles(roi, circles, HOUGH_GRADIENT, 1,
                        roi.rows/8,  // 最小圆心距离
                        40, 5,       // Canny阈值，累加器阈值
                        rmin, rmax); // 最小和最大半径

        // 找到最佳匹配的圆
        double bestFit = numeric_limits<double>::max();
        Vec3f bestCircle;
        bool found = false;

        for(const auto& circle : circles) 
        {
            float radius = circle[2];
            float bias = abs(radius - r);

            if(bias < bestFit) 
            {
                bestFit = bias;
                bestCircle = circle;
                found = true;
            }
        }

        if(found) 
        {
            // 调整圆心坐标到原图坐标系
            bestCircle[0] += searchAreas[i].x;
            bestCircle[1] += searchAreas[i].y;
            detectedCircles.push_back(bestCircle);
            detectedCenters.push_back(Point2f(bestCircle[0], bestCircle[1]));
            detectedRadii.push_back(bestCircle[2]);
            detectedWorldCoords.push_back(worldCoords[i]);
        }
    }

    // 亚像素精度优化圆心坐标 - 使用图像矩计算质心
    for (size_t i = 0; i < detectedCenters.size(); i++) 
    {
        // 根据圆的半径动态定义ROI区域 (以检测到的圆心为中心，2r*2r像素区域) 
        radius = static_cast<int>(detectedRadii[i]);
        roi_size = 2 * radius; // ROI大小为2r*2r
        x = max(0, static_cast<int>(detectedCenters[i].x - radius));
        y = max(0, static_cast<int>(detectedCenters[i].y - radius));
        int roi_width = min(roi_size, grayImage.cols - x);
        int roi_height = min(roi_size, grayImage.rows - y);
        
        if (roi_width > 0 && roi_height > 0) 
        {
            Mat roi = grayImage(Rect(x, y, roi_width, roi_height));
            
            // 计算图像矩
            m = moments(roi, true);
            
            if (m.m00 != 0) 
            {
                // 计算质心 (相对于ROI的坐标) 
                cx = m.m10 / m.m00;
                cy = m.m01 / m.m00;
                
                // 更新圆心坐标 (转换回原图坐标系) 
                detectedCenters[i].x = x + cx;
                detectedCenters[i].y = y + cy;
            }
        }
    }

    // 存储优化后的结果并绘制
    for (size_t i = 0; i < detectedCenters.size(); i++) 
    {
        WorldCoord.append(QPointF(detectedWorldCoords[i].x, detectedWorldCoords[i].y));
        PixelCoord.append(QPointF(detectedCenters[i].x, detectedCenters[i].y));

        // 绘制结果
        circle(result, detectedCenters[i], detectedRadii[i], Scalar(0, 255, 0), 10);
        circle(result, detectedCenters[i], 10, Scalar(0, 0, 255), -1);
    }

    // 保存结果图像
    imwrite("/home/orangepi/Desktop/VisualRobot_Local/Img/circle_detected.jpg", result);

    return detectedCenters.empty() ? 1 : 0;
}

// 从文件读取变换矩阵
// 参数:
//   filename - 矩阵文件路径
// 返回值:
//   成功返回变换矩阵，失败抛出异常
Matrix3d ReadTransformationMatrix(const string& filename)
{
    // 变量定义
    ifstream file;     // 文件流对象
    Matrix3d matrix;   // 变换矩阵
    string line;       // 文件行内容
    istringstream iss; // 字符串流用于解析
    
    file.open(filename);
    if (!file)
    {
        throw runtime_error("Cannot open file: " + filename);
    }

    // 逐行读取文件内容
    for (int i = 0; i < 3; ++i)
    {
        string line;
        if (!getline(file, line))
        {
            throw runtime_error("Unexpected end of file: " + filename);
        }

        istringstream iss(line);
        for (int j = 0; j < 3; ++j)
        {
            if (!(iss >> matrix(i, j)))
            {
                throw runtime_error("Error parsing matrix data in file: " + filename);
            }
        }
    }

    return matrix;
}

// OpenCV版本的矩形检测算法
// 参数:
//   imgPath - 输入图像路径
//   Row - 输出参数，存储矩形角点的行坐标
//   Col - 输出参数，存储矩形角点的列坐标
// 返回值:
//   成功返回0，失败返回1
int DetectRectangleOpenCV(const string& imgPath, vector<double>& Row, vector<double>& Col)
{
    // 变量定义
    Mat image, grayImage, binaryImage, smoothedImage, edges; // 图像处理变量
    vector<vector<Point>> contours;                          // 轮廓列表
    vector<Vec4i> hierarchy;                                 // 轮廓层级
    vector<vector<Point>> longContours;                      // 长轮廓列表
    vector<Point> rectContour;                               // 矩形轮廓
    double maxArea;                                          // 最大面积
    RotatedRect rect;                                        // 旋转矩形
    vector<Point2f> boxPoints;                               // 矩形角点
    vector<Point> intPoints;                                 // 整数角点
    vector<Point> orderedPoints;                             // 排序后的角点
    vector<Point2f> corners;                                 // 角点用于亚像素优化
    Size winSize, zeroZone;                                  // 亚像素优化参数
    TermCriteria criteria;                                   // 终止条件
    Mat result;                                              // 结果图像
    
    // 清空输入向量
    Row.clear();
    Col.clear();

    // 读取图像
    image = imread(imgPath);
    if (image.empty()) 
    {
        qDebug() << "Error: Cannot read image file";
        return 1;
    }

    // 转换为灰度图
    cvtColor(image, grayImage, COLOR_BGR2GRAY);

    // 二值化
    threshold(grayImage, binaryImage, 128, 255, THRESH_BINARY);

    // 高斯滤波平滑处理
    GaussianBlur(binaryImage, smoothedImage, Size(3, 3), 0);

    // Canny边缘检测
    Canny(smoothedImage, edges, 20, 40);

    // 查找轮廓
    findContours(edges, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    // 过滤短轮廓
    for(const auto& contour : contours) 
    {
        if(arcLength(contour, true) > 50) 
        {
            longContours.push_back(contour);
        }
    }

    // 寻找最大的矩形轮廓
    maxArea = 0;
    for(const auto& contour : longContours) 
    {
        double area = contourArea(contour);
        if(area > maxArea) 
        {
            rect = minAreaRect(contour);
            boxPoints.resize(4);
            rect.points(boxPoints.data());
            
            // 转换为整数点
            intPoints.clear();
            for(const auto& pt : boxPoints) 
            {
                intPoints.push_back(Point(static_cast<int>(pt.x), static_cast<int>(pt.y)));
            }
            
            maxArea = area;
            rectContour = intPoints;
        }
    }

    // 如果找到矩形
    if(!rectContour.empty()) 
    {
        // 按照顺时针顺序排序角点
        vector<Point> orderedPoints = rectContour;
        // 按y坐标排序 (从上到下) 
        sort(orderedPoints.begin(), orderedPoints.end(), [](const Point& a, const Point& b) { return a.y < b.y; });
        
        // 上面的两个点按x坐标排序 (从左到右) 
        if(orderedPoints[0].x > orderedPoints[1].x) 
        {
            swap(orderedPoints[0], orderedPoints[1]);
        }
        
        // 下面的两个点按x坐标排序 (从左到右) 
        if(orderedPoints[2].x > orderedPoints[3].x) 
        {
            swap(orderedPoints[2], orderedPoints[3]);
        }

        // 存储角点坐标
        for(const auto& pt : orderedPoints) 
        {
            Row.push_back(pt.y);
            Col.push_back(pt.x);
        }

        // 亚像素精度优化
        if (!rectContour.empty()) 
        {
            // 将角点转换为Point2f格式用于亚像素优化
            corners.clear();
            for (int i = 0; i < 4; i++) 
            {
                corners.push_back(Point2f(Col[i], Row[i]));
            }
            
            // 配置亚像素优化参数
            winSize = Size(5, 5);
            zeroZone = Size(-1, -1);
            criteria = TermCriteria(TermCriteria::EPS + TermCriteria::MAX_ITER, 40, 0.001);
            
            // 执行亚像素角点优化
            cornerSubPix(grayImage, corners, winSize, zeroZone, criteria);
            
            // 更新优化后的角点坐标
            for (int i = 0; i < 4; i++) 
            {
                Row[i] = corners[i].y;
                Col[i] = corners[i].x;
            }
        }

        // 可视化结果
        result = image.clone();

        // 然后绘制红色的1像素精确矩形轮廓
        for(int i = 0; i < 4; i++) 
        {
            line(result, rectContour[i], rectContour[(i+1)%4], Scalar(0, 255, 0), 1);
        }

        // 绘制角点 - 首先在精确的角点位置绘制红色的8像素为半径的角点圆
        for(int i = 0; i < 4; i++) 
        {
            circle(result, Point(Col[i], Row[i]), 8, Scalar(0, 0, 255), -1); // 填充红色圆，半径为8像素
        }

        // 然后绘制绿色的1像素为半径的角点圆
        for(int i = 0; i < 4; i++) 
        {
            circle(result, Point(Col[i], Row[i]), 1, Scalar(0, 255, 0), -1); // 填充绿色圆，半径为1像素
        }

        // 保存结果图像
        imwrite("../detectedImg.jpg", result);
        
        return 0;
    }

    return 1;
}


// 处理图像并计算尺寸的函数
// 参数:
//   input - 输入图像
//   params - 处理参数
//   bias - 比例偏差
// 返回值:
//   包含宽度、高度、角度和图像的结果结构体
Result CalculateLength(const Mat& input, const Params& params, double bias)
{
    // 变量定义
    Result result;                            // 结果结构体
    Mat source, binary, colorImage;           // 图像处理变量
    int k;                                    // 滤波核大小
    static Mat kernel;                        // 形态学核
    vector<vector<Point>> contours;           // 轮廓列表
    vector<Vec4i> hierarchy;                  // 轮廓层级
    vector<vector<Point>> preservedContours;  // 保留的轮廓
    vector<vector<Point>> filteredContours;   // 过滤的轮廓
    int thickness;                            // 绘制线宽
    RotatedRect rotatedRect;                  // 旋转矩形
    Size2f rotatedSize;                       // 旋转矩形尺寸
    float spring_length, spring_width, angle; // 弹簧尺寸和角度
    Point2f vertices[4];                      // 矩形顶点
    int thickBorder;                          // 边框厚度
    
    if (input.empty()) 
    {
        cerr << "图像读取失败" << endl;
        return result;
    }

    // 检查通道数，如果需要则转换为灰度图
    if (input.channels() > 1) 
    {
        cvtColor(input, source, COLOR_BGR2GRAY);
    } 
    else 
    {
        source = input;
    }

    // 多阶段滤波
    if (params.blurK >= 3) 
    {
        k = (params.blurK % 2 == 0) ? params.blurK - 1 : params.blurK;
        if (k >= 3) 
        {
            Mat dst;
            bilateralFilter(source, dst, 5, 30, 2);  // 双边滤波
            GaussianBlur(dst, source, Size(k, k), 2.0, 2.0); // 高斯模糊
        }
    }

    // 预计算核
    kernel = getStructuringElement(MORPH_RECT, Size(3, 3));

    // 阈值处理
    binary = source > params.thresh;
    binary = 255 - binary;

    // 形态学操作
    morphologyEx(binary, binary, MORPH_DILATE, kernel);  // 仅膨胀操作

    // 查找轮廓
    findContours(binary, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    // 按面积过滤轮廓
    for (const auto& contour : contours) 
    {
        if (contourArea(contour) < params.areaMin) 
        {
            filteredContours.push_back(contour);
        } 
        else 
        {
            preservedContours.push_back(contour);
        }
    }

    // 查找最大面积的轮廓
    auto max_it = max_element(preservedContours.begin(), preservedContours.end(),
                         [](const vector<Point>& a, const vector<Point>& b) {
                             return contourArea(a) < contourArea(b);
                         });

    // 基于图像宽度计算线宽
    thickness = round(source.cols * 0.002);

    // 转换为BGR用于彩色绘图
    cvtColor(source, colorImage, COLOR_GRAY2BGR);

    if (max_it != preservedContours.end()) 
    {
        // 获取旋转矩形
        rotatedRect = minAreaRect(*max_it);
        rotatedSize = rotatedRect.size;

        spring_length = max(rotatedSize.width, rotatedSize.height);
        spring_width = min(rotatedSize.width, rotatedSize.height);
        result.widths.push_back(spring_width*bias);
        result.heights.push_back(spring_length*bias);

        // 用绿色绘制边界框
        rotatedRect.points(vertices);
        thickBorder = static_cast<int>(thickness * 1);

        for (int i = 0; i < 4; i++) 
        {
            line(colorImage, vertices[i], vertices[(i+1)%4], Scalar(0, 255, 0),  thickBorder);
        }

        cout << "物件长度: " << spring_length*bias << ", 物件宽度: " << spring_width*bias << endl;
        angle = rotatedRect.angle;
        cout << "检测角度: " << angle << "°" << endl;
        result.angles.push_back(angle);
    } 
    else 
    {
        result.widths.push_back(0);
        result.heights.push_back(0);
        result.angles.push_back(0);
        cerr << "未找到有效轮廓" << endl;
    }

    result.image = colorImage;
    return result;
}

// 基于连通域的多目标检长函数
// 参数:
//   input - 输入图像
//   params - 处理参数
//   bias - 比例偏差
// 返回值:
//   包含所有目标宽度、高度、角度和图像的结果结构体
Result CalculateLengthMultiTarget(const Mat& input, const Params& params, double bias)
{
    // 变量定义
    Result result;                            // 结果结构体
    Mat source, binary, colorImage;           // 图像处理变量
    int k;                                    // 滤波核大小
    static Mat kernel;                        // 形态学核
    vector<vector<Point>> contours;           // 轮廓列表
    vector<Vec4i> hierarchy;                  // 轮廓层级
    int thickness;                            // 绘制线宽
    RotatedRect rotatedRect;                  // 旋转矩形
    Size2f rotatedSize;                       // 旋转矩形尺寸
    float spring_length, spring_width, angle; // 弹簧尺寸和角度
    Point2f vertices[4];                      // 矩形顶点
    int thickBorder;                          // 边框厚度
    int targetIndex;                          // 目标序号
    size_t i;                                 // 循环索引

    if (input.empty())
    {
        cerr << "图像读取失败" << endl;
        return result;
    }

    // 检查通道数，如果需要则转换为灰度图
    if (input.channels() > 1)
    {
        cvtColor(input, source, COLOR_BGR2GRAY);
    }
    else
    {
        source = input;
    }

    // 多阶段滤波
    if (params.blurK >= 3)
    {
        k = (params.blurK % 2 == 0) ? params.blurK - 1 : params.blurK;
        if (k >= 3)
        {
            Mat dst;
            bilateralFilter(source, dst, 5, 30, 2);  // 双边滤波
            GaussianBlur(dst, source, Size(k, k), 2.0, 2.0); // 高斯模糊
        }
    }

    // 预计算核
    kernel = getStructuringElement(MORPH_RECT, Size(3, 3));

    // 阈值处理
    binary = source > params.thresh;
    binary = 255 - binary;

    // 形态学操作
    morphologyEx(binary, binary, MORPH_DILATE, kernel);  // 仅膨胀操作

    // 查找轮廓 - 仅搜索最外层轮廓
    findContours(binary, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    // 转换为BGR用于彩色绘图
    cvtColor(source, colorImage, COLOR_GRAY2BGR);

    // 基于图像宽度计算线宽
    thickness = round(source.cols * 0.002);
    thickBorder = static_cast<int>(thickness * 1);

    // 确保Img目录存在
    CreateDirectory("../Img");

    // 创建Canny边缘检测结果图像（与原图相同尺寸）
    Mat cannyResult = Mat::zeros(input.size(), CV_8UC1);

    // 存储轮廓信息和对应的ROI区域
    struct ContourInfo {
        vector<Point> contour;
        RotatedRect rect;
        Rect boundingBox;
        int targetIndex;
    };
    vector<ContourInfo> contourInfos;

    // 第一阶段：处理所有轮廓，收集信息并执行Canny边缘检测
    targetIndex = 1;
    for (i = 0; i < contours.size(); i++)
    {
        const auto& contour = contours[i];

        // 面积过滤
        if (contourArea(contour) < params.areaMin)
        {
            continue;
        }

        // 计算最小外接矩形
        rotatedRect = minAreaRect(contour);

        // 获取ROI区域
        Rect boundingBox = rotatedRect.boundingRect();
        boundingBox = boundingBox & Rect(0, 0, input.cols, input.rows);

        if (!boundingBox.empty() && boundingBox.width > 0 && boundingBox.height > 0)
        {
            // 保存目标图像
            Mat targetImage = input(boundingBox).clone();
            string filename = "../Img/object0" + to_string(targetIndex) + ".jpg";
            bool saveSuccess = imwrite(filename, targetImage);
            if (saveSuccess)
            {
                cout << "目标" << targetIndex << "图像已保存: " << filename << endl;
            }
            else
            {
                cerr << "保存目标" << targetIndex << "图像失败: " << filename << endl;
            }

            // 在ROI区域内进行Canny边缘检测
            Mat roiImage;
            if (binary.channels() > 1)
            {
                cvtColor(binary(boundingBox), roiImage, COLOR_BGR2GRAY);
            }
            else
            {
                roiImage = binary(boundingBox).clone();
            }

            Mat edges;
            Canny(roiImage, edges, 100, 200, 3);

            // 将边缘检测结果复制到Canny结果图像的对应位置
            edges.copyTo(cannyResult(boundingBox));

            cout << "目标" << targetIndex << "Canny边缘检测完成，ROI尺寸: " << boundingBox.width << "x" << boundingBox.height << endl;

            // 保存轮廓信息
            ContourInfo info;
            info.contour = contour;
            info.rect = rotatedRect;
            info.boundingBox = boundingBox;
            info.targetIndex = targetIndex;
            contourInfos.push_back(info);

            targetIndex++;
        }
    }

    // 在Canny边缘检测结果基础上处理边缘
    if (!cannyResult.empty())
    {
        // 创建轮廓掩码，用于排除与轮廓近似的边缘
        Mat contourMask = Mat::zeros(input.size(), CV_8UC1);

        // 绘制所有轮廓到掩码上（白色）
        for (i = 0; i < contours.size(); i++)
        {
            const auto& contour = contours[i];
            if (contourArea(contour) >= params.areaMin)
            {
                // 绘制轮廓到掩码，线宽稍微加粗以覆盖近似边缘
                drawContours(contourMask, contours, i, Scalar(255), 3);
            }
        }

        // 对轮廓掩码进行膨胀操作，确保覆盖所有近似边缘
        Mat dilatedContourMask;
        dilate(contourMask, dilatedContourMask, kernel, Point(-1, -1), 2);

        // 从Canny结果中排除与轮廓近似的边缘
        Mat filteredEdges;
        bitwise_and(cannyResult, ~dilatedContourMask, filteredEdges);

        // 查找剩余边缘的轮廓
        vector<vector<Point>> edgeContours;
        vector<Vec4i> edgeHierarchy;
        findContours(filteredEdges, edgeContours, edgeHierarchy, RETR_LIST, CHAIN_APPROX_SIMPLE);

        // 对每个边缘轮廓进行平滑拟合并绘制到原图
        for (const auto& edgeContour : edgeContours)
        {
            // 过滤掉太小的边缘
            if (contourArea(edgeContour) < 1)
            {
                continue;
            }

            // 多边形近似 - 平滑拟合
            vector<Point> approxCurve;
            double epsilon = 0.005 * arcLength(edgeContour, true);
            approxPolyDP(edgeContour, approxCurve, epsilon, true);

            // 如果近似后点数太少，使用原始轮廓
            if (approxCurve.size() < 3)
            {
                approxCurve = edgeContour;
            }

            // 绘制平滑的红色边缘到原图
            for (size_t j = 0; j < approxCurve.size(); j++)
            {
                size_t next = (j + 1) % approxCurve.size();
                line(colorImage, approxCurve[j], approxCurve[next], Scalar(0, 0, 255), 2);
            }
        }

        cout << "已处理 " << edgeContours.size() << " 个边缘轮廓，并用红色平滑线条显示" << endl;

        // 第二阶段：绘制轮廓边界框和序号，根据是否存在非轮廓边缘决定颜色
        for (const auto& info : contourInfos)
        {
            // 检查该轮廓ROI范围内是否存在非轮廓近似的边缘
            Mat roiFilteredEdges = filteredEdges(info.boundingBox);
            int nonZeroCount = countNonZero(roiFilteredEdges);

            // 决定颜色：如果存在非轮廓边缘，使用红色；否则使用绿色
            Scalar contourColor;
            if (nonZeroCount > 0)
            {
                contourColor = Scalar(0, 0, 255);  // 红色 - 存在非轮廓边缘
                cout << "目标" << info.targetIndex << " ROI内检测到非轮廓边缘，数量: " << nonZeroCount << endl;
            }
            else
            {
                contourColor = Scalar(0, 255, 0);  // 绿色 - 无非轮廓边缘
            }

            // 绘制边界框
            info.rect.points(vertices);
            for (int k = 0; k < 4; k++)
            {
                line(colorImage, vertices[k], vertices[(k+1)%4], contourColor, thickBorder);
            }

            // 在矩形左上角绘制序号
            Point2f rectCenter = info.rect.center;
            Size2f rectSize = info.rect.size;
            Point textPosition;

            // 计算文本位置（矩形左上角）
            if (info.rect.angle < -45)
            {
                textPosition = Point(static_cast<int>(rectCenter.x - rectSize.width/2), static_cast<int>(rectCenter.y - rectSize.height/2));
            }
            else
            {
                textPosition = Point(static_cast<int>(rectCenter.x - rectSize.height/2), static_cast<int>(rectCenter.y - rectSize.width/2));
            }

            // 确保文本位置在图像范围内
            textPosition.x = max(5, textPosition.x);
            textPosition.y = max(20, textPosition.y);

            // 绘制序号文本（使用与边界框相同的颜色）
            putText(colorImage, to_string(info.targetIndex), textPosition, FONT_HERSHEY_SIMPLEX, 4, contourColor, 10);

            // 计算尺寸并保存结果
            spring_length = max(info.rect.size.width, info.rect.size.height);
            spring_width = min(info.rect.size.width, info.rect.size.height);
            angle = info.rect.angle;

            result.widths.push_back(spring_width * bias);
            result.heights.push_back(spring_length * bias);
            result.angles.push_back(angle);
        }
    }
    else
    {
        // 如果没有Canny结果，使用默认绿色绘制所有轮廓
        for (const auto& info : contourInfos)
        {
            // 绘制边界框
            info.rect.points(vertices);
            for (int k = 0; k < 4; k++)
            {
                line(colorImage, vertices[k], vertices[(k+1)%4], Scalar(0, 255, 0), thickBorder);
            }

            // 在矩形左上角绘制序号
            Point2f rectCenter = info.rect.center;
            Size2f rectSize = info.rect.size;
            Point textPosition;

            // 计算文本位置（矩形左上角）
            if (info.rect.angle < -45)
            {
                textPosition = Point(static_cast<int>(rectCenter.x - rectSize.width/2), static_cast<int>(rectCenter.y - rectSize.height/2));
            }
            else
            {
                textPosition = Point(static_cast<int>(rectCenter.x - rectSize.height/2), static_cast<int>(rectCenter.y - rectSize.width/2));
            }

            // 确保文本位置在图像范围内
            textPosition.x = max(5, textPosition.x);
            textPosition.y = max(20, textPosition.y);

            // 绘制序号文本
            putText(colorImage, to_string(info.targetIndex), textPosition, FONT_HERSHEY_SIMPLEX, 4, Scalar(0, 255, 0), 10);

            // 计算尺寸并保存结果
            spring_length = max(info.rect.size.width, info.rect.size.height);
            spring_width = min(info.rect.size.width, info.rect.size.height);
            angle = info.rect.angle;

            result.widths.push_back(spring_width * bias);
            result.heights.push_back(spring_length * bias);
            result.angles.push_back(angle);
        }
    }

    if (result.widths.empty())
    {
        cerr << "未找到有效轮廓" << endl;
    }

    result.image = colorImage;
    return result;
}
