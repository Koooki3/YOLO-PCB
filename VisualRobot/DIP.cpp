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
#include <opencv2/highgui.hpp>
#include <vector>
#include <algorithm>

#define ERROR 2
#define WARNNING 1
#define INFO 0

using namespace Eigen;
using namespace std;
using namespace cv;

// 创建目录的辅助函数
bool createDirectory(const string& path)
{
    QDir dir(QString::fromStdString(path));
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
int TransMatrix(const QVector<QPointF>& WorldCoord, const QVector<QPointF>& PixelCoord, Matrix3d& matrix, const string& filename = "")
{
    // 参数校验
    const int pointCount = WorldCoord.count();
    if (WorldCoord.count() != PixelCoord.count())
    {
        cerr << "错误: 世界坐标与像素坐标数量不匹配" << endl;
        return 1; // 错误码1: 坐标数量不匹配
    }

    if (pointCount < 3)
    {
        cerr << "错误: 至少需要3个对应点" << endl;
        return 2; // 错误码2: 点数不足
    }

    try
    {
        // 预分配内存并避免重复计算
        MatrixXd A = MatrixXd::Zero(2 * pointCount, 6);
        VectorXd b = VectorXd::Zero(2 * pointCount);

        // 使用更高效的矩阵填充方式
        for (int i = 0; i < pointCount; ++i)
        {
            const double u = PixelCoord[i].x();
            const double v = PixelCoord[i].y();
            const double x = WorldCoord[i].x();
            const double y = WorldCoord[i].y();

            const int row1 = 2 * i;
            const int row2 = 2 * i + 1;

            // x坐标方程: x = a*u + b*v + c
            A(row1, 0) = u;
            A(row1, 1) = v;
            A(row1, 2) = 1.0;
            b(row1) = x;

            // y坐标方程: y = d*u + e*v + f
            A(row2, 3) = u;
            A(row2, 4) = v;
            A(row2, 5) = 1.0;
            b(row2) = y;
        }

        // 精度优化1: 数据归一化处理（提高数值稳定性）
        Vector2d pixel_mean = Vector2d::Zero();
        Vector2d world_mean = Vector2d::Zero();

        for (int i = 0; i < pointCount; ++i) 
        {
            pixel_mean += Vector2d(PixelCoord[i].x(), PixelCoord[i].y());
            world_mean += Vector2d(WorldCoord[i].x(), WorldCoord[i].y());
        }
        pixel_mean /= pointCount;
        world_mean /= pointCount;

        double pixel_scale = 0.0;
        double world_scale = 0.0;
        for (int i = 0; i < pointCount; ++i) 
        {
            pixel_scale += (Vector2d(PixelCoord[i].x(), PixelCoord[i].y()) - pixel_mean).norm();
            world_scale += (Vector2d(WorldCoord[i].x(), WorldCoord[i].y()) - world_mean).norm();
        }
        pixel_scale /= pointCount;
        world_scale /= pointCount;

        // 高级精度优化1: 加权最小二乘法
        // 根据点的质量分配权重（距离中心越近的点权重越高）
        VectorXd weights = VectorXd::Ones(2 * pointCount);
        Vector2d pixel_center = pixel_mean;
        for (int i = 0; i < pointCount; ++i) 
        {
            Vector2d pixel_vec(PixelCoord[i].x(), PixelCoord[i].y());
            double distance = (pixel_vec - pixel_center).norm();
            double weight = exp(-distance * distance / (2.0 * pixel_scale * pixel_scale));
            weights(2*i) = weight;
            weights(2*i+1) = weight;
        }

        // 构建加权矩阵
        MatrixXd W = weights.asDiagonal();
        MatrixXd ATWA = A.transpose() * W * A;
        VectorXd ATWb = A.transpose() * W * b;

        // 高级精度优化2: 使用更稳定的求解方法
        VectorXd x = ATWA.completeOrthogonalDecomposition().solve(ATWb);

        // 高级精度优化3: 多次迭代精化
        const int max_refinement_iterations = 3;
        double prev_error = numeric_limits<double>::max();

        for (int iter = 0; iter < max_refinement_iterations; ++iter) 
        {
            VectorXd residual = b - A * x;
            double current_error = residual.norm();

            // 如果误差不再显著减小，停止迭代
            if (abs(prev_error - current_error) < 1e-12 * prev_error) 
            {
                break;
            }
            prev_error = current_error;

            // 求解修正量
            VectorXd correction = ATWA.completeOrthogonalDecomposition().solve(A.transpose() * W * residual);
            x += correction;
        }

        // 高级精度优化4: RANSAC异常点检测（可选）
        // 可以添加RANSAC来检测和移除异常点，进一步提高精度

        // 计算最终误差并输出
        VectorXd final_residual = b - A * x;
        double final_error = final_residual.norm();
        cout << "最终残差范数: " << scientific << setprecision(6) << final_error << endl;

        // 输出每个点的误差
        cout << "各点误差分析:" << endl;
        for (int i = 0; i < pointCount; ++i) 
        {
            Vector2d pixel_err(final_residual(2*i), final_residual(2*i+1));
            cout << "点 " << i << ": 误差 = " << pixel_err.norm() << " pixels" << endl;
        }

        // 构建变换矩阵
        matrix << x(0), x(1), x(2),
                  x(3), x(4), x(5),
                  0.0,  0.0,  1.0;

        // 输出矩阵信息
        cout << "变换矩阵:" << endl;
        cout << fixed << setprecision(10);
        cout << matrix << endl;

        // 验证变换矩阵
        cout << "验证变换结果:" << endl;
        for (int i = 0; i < pointCount; ++i)
        {
            Vector3d pixel(PixelCoord[i].x(), PixelCoord[i].y(), 1.0);
            Vector3d world = matrix * pixel;

            double error_x = abs(world[0] - WorldCoord[i].x());
            double error_y = abs(world[1] - WorldCoord[i].y());

            cout << "像素坐标: (" << pixel[0] << ", " << pixel[1] << ") -> ";
            cout << "计算的世界坐标: (" << world[0] << ", " << world[1] << ") ";
            cout << "实际的世界坐标: (" << WorldCoord[i].x() << ", " << WorldCoord[i].y() << ")" << endl;
            cout << " 误差: (" << error_x << ", " << error_y << ")" << endl;
        }

        // 如果提供了文件名，保存矩阵到文件
        if (!filename.empty())
        {
            // 使用相对路径 "../matrix.bin"
            string filePath = "../matrix.bin";

            // 保存矩阵到文件
            ofstream fout(filePath, ios::binary);
            if (fout.is_open())
            {
                fout << fixed << setprecision(10) << matrix << endl;
                fout.close();
                cout << "矩阵保存成功: " << filePath << endl;
            }
            else
            {
                cerr << "错误: 无法打开文件 " << filePath << endl;
                return 4; // 错误码4: 文件打开失败
            }
        }

        return 0; // 成功

    }
    catch (const exception& e)
    {
        cerr << "错误: " << e.what() << endl;
        return 6; // 错误码6: 计算过程中发生异常
    }
}


// OpenCV版本的圆形检测算法
int getCoords_opencv(QVector<QPointF>& WorldCoord, QVector<QPointF>& PixelCoord, double size = 100.0)
{
    // 读取图像
    Mat image = imread("/home/orangepi/Desktop/VisualRobot_Local/Img/capture.jpg");
    if(image.empty()) 
    {
        qDebug() << "Error: Cannot read image file";
        return 1;
    }

    // 获取图像尺寸
    int height = image.rows;
    int width = image.cols;

    // 计算参数（基于原始代码中的比例）
    double r = (height * 316.0) / 2182.0;
    double dis_col = (width * 1049.0) / 2734.0;
    double dis_row = (height * 774.0) / 2182.0;
    double rmin = r - 10;
    double rmax = r + 10;

    // 定义9个搜索区域
    vector<Rect> searchAreas;
    vector<Point2f> worldCoords;
    
    // 计算世界坐标
    double world_dis_col = 1049.0 * size / 2734.0;
    double world_dis_row = 774.0 * size / 2734.0;

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
    Mat grayImage;
    cvtColor(image, grayImage, COLOR_BGR2GRAY);

    // 二值化
    Mat binaryImage;
    threshold(grayImage, binaryImage, 200, 255, THRESH_BINARY);

    // 在每个区域中检测圆
    vector<Vec3f> detectedCircles;
    Mat result = image.clone();
    vector<Point2f> detectedCenters; // 存储检测到的圆心
    vector<float> detectedRadii; // 存储检测到的半径
    vector<Point2f> detectedWorldCoords; // 存储对应的世界坐标

    for(size_t i = 0; i < searchAreas.size(); i++) 
    {
        Mat roi = binaryImage(searchAreas[i]);
        vector<Vec3f> circles;

        // 使用Hough圆变换检测圆
        HoughCircles(roi, circles, HOUGH_GRADIENT, 1,
                        roi.rows/8,  // 最小圆心距离
                        40, 5,     // Canny阈值，累加器阈值
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
        // 根据圆的半径动态定义ROI区域（以检测到的圆心为中心，2r*2r像素区域）
        int radius = static_cast<int>(detectedRadii[i]);
        int roi_size = 2 * radius; // ROI大小为2r*2r
        int x = max(0, static_cast<int>(detectedCenters[i].x - radius));
        int y = max(0, static_cast<int>(detectedCenters[i].y - radius));
        int width = min(roi_size, grayImage.cols - x);
        int height = min(roi_size, grayImage.rows - y);
        
        if (width > 0 && height > 0) 
        {
            Mat roi = grayImage(Rect(x, y, width, height));
            
            // 计算图像矩
            Moments m = moments(roi, true);
            
            if (m.m00 != 0) 
            {
                // 计算质心（相对于ROI的坐标）
                double cx = m.m10 / m.m00;
                double cy = m.m01 / m.m00;
                
                // 更新圆心坐标（转换回原图坐标系）
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

Matrix3d readTransformationMatrix(const string& filename)
{
    ifstream file(filename);
    if (!file)
    {
        throw runtime_error("Cannot open file: " + filename);
    }

    Matrix3d matrix;

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
int Algorithm_opencv(const string& imgPath, vector<double>& Row, vector<double>& Col)
{
    // 清空输入向量
    Row.clear();
    Col.clear();

    // 读取图像
    Mat image = imread(imgPath);
    if (image.empty()) 
    {
        qDebug() << "Error: Cannot read image file";
        return 1;
    }

    // 转换为灰度图
    Mat grayImage;
    cvtColor(image, grayImage, COLOR_BGR2GRAY);

    // 二值化
    Mat binaryImage;
    threshold(grayImage, binaryImage, 128, 255, THRESH_BINARY);

    // 高斯滤波平滑处理
    Mat smoothedImage;
    GaussianBlur(binaryImage, smoothedImage, Size(3, 3), 0);

    // Canny边缘检测
    Mat edges;
    Canny(smoothedImage, edges, 20, 40);

    // 查找轮廓
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    findContours(edges, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    // 过滤短轮廓
    vector<vector<Point>> longContours;
    for(const auto& contour : contours) 
    {
        if(arcLength(contour, true) > 50) 
        {
            longContours.push_back(contour);
        }
    }

    // 寻找最大的矩形轮廓
    vector<Point> rectContour;
    double maxArea = 0;
    for(const auto& contour : longContours) 
    {
        double area = contourArea(contour);
        if(area > maxArea) 
        {
            RotatedRect rect = minAreaRect(contour);
            vector<Point2f> boxPoints(4);
            rect.points(boxPoints.data());
            
            // 转换为整数点
            vector<Point> intPoints;
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
        // 按y坐标排序（从上到下）
        sort(orderedPoints.begin(), orderedPoints.end(), 
             [](const Point& a, const Point& b) { return a.y < b.y; });
        
        // 上面的两个点按x坐标排序（从左到右）
        if(orderedPoints[0].x > orderedPoints[1].x) 
        {
            swap(orderedPoints[0], orderedPoints[1]);
        }
        
        // 下面的两个点按x坐标排序（从左到右）
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
            vector<Point2f> corners;
            for (int i = 0; i < 4; i++) 
            {
                corners.push_back(Point2f(Col[i], Row[i]));
            }
            
            // 配置亚像素优化参数
            Size winSize = Size(5, 5);
            Size zeroZone = Size(-1, -1);
            TermCriteria criteria = TermCriteria(TermCriteria::EPS + TermCriteria::MAX_ITER, 40, 0.001);
            
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
        Mat result = image.clone();

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
Result calculateLength(const Mat& input, const Params& params, double bias) {
    Result result;

    if (input.empty()) 
    {
        cerr << "图像读取失败" << endl;
        return result;
    }

    Mat source;

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
        int k = (params.blurK % 2 == 0) ? params.blurK - 1 : params.blurK;
        if (k >= 3) 
        {
            Mat dst;
            bilateralFilter(source, dst, 5, 30, 2);  // 双边滤波
            GaussianBlur(dst, source, Size(k, k), 2.0, 2.0); // 高斯模糊
        }
    }

    // 预计算核
    static Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));

    // 阈值处理
    Mat binary;
    binary = source > params.thresh;
    binary = 255 - binary;

    // 形态学操作
    morphologyEx(binary, binary, MORPH_DILATE, kernel);  // 仅膨胀操作

    // 查找轮廓
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    vector<vector<Point>> preservedContours;
    vector<vector<Point>> filteredContours;

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
    const int thickness = round(source.cols * 0.002);

    // 转换为BGR用于彩色绘图
    Mat colorImage;
    cvtColor(source, colorImage, COLOR_GRAY2BGR);

    if (max_it != preservedContours.end()) 
    {
        // 获取旋转矩形
        RotatedRect rotatedRect = minAreaRect(*max_it);
        Size2f rotatedSize = rotatedRect.size;

        float spring_length = max(rotatedSize.width, rotatedSize.height);
        float spring_width = min(rotatedSize.width, rotatedSize.height);
        result.widths.push_back(spring_width*bias);
        result.heights.push_back(spring_length*bias);

        // 用绿色绘制边界框
        Point2f vertices[4];
        rotatedRect.points(vertices);
        const int thickBorder = static_cast<int>(thickness * 1);

        for (int i = 0; i < 4; i++) 
        {
            line(colorImage,
                 vertices[i],
                 vertices[(i+1)%4],
                 Scalar(0, 255, 0),  // 绿色
                 thickBorder);
        }

        cout << "物件长度: " << spring_length*bias << ", 物件宽度: " << spring_width*bias << endl;
        float angle = rotatedRect.angle;
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
