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
#include <halconcpp/HalconCpp.h>
#include <Halcon.h>
#include <halconcpp/HDevThread.h>
#include <cmath>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#define ERROR 2
#define WARNNING 1
#define INFO 0

using namespace HalconCpp;
using namespace Eigen;
using namespace std;

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
    if (WorldCoord.count() != PixelCoord.count())
    {
        cerr << "错误: 世界坐标与像素坐标数量不匹配" << endl;
        return 1; // 错误码1: 坐标数量不匹配
    }

    const int pointCount = WorldCoord.count();
    if (pointCount < 3)
    {
        cerr << "错误: 至少需要3个对应点" << endl;
        return 2; // 错误码2: 点数不足
    }

    try
    {
        // 转换像素坐标到Eigen矩阵（n×3齐次坐标）
        Eigen::MatrixXd PixelCoord1(pointCount, 3);
        for(int i = 0; i < pointCount; ++i)
        {
            PixelCoord1(i, 0) = PixelCoord[i].x();
            PixelCoord1(i, 1) = PixelCoord[i].y();
            PixelCoord1(i, 2) = 1.0;
        }

        // 转换世界坐标到Eigen矩阵（n×3齐次坐标）
        Eigen::MatrixXd WorldCoord1(pointCount, 3);
        for(int i = 0; i < pointCount; ++i)
        {
            WorldCoord1(i, 0) = WorldCoord[i].x();       // X坐标
            WorldCoord1(i, 1) = WorldCoord[i].y();       // Y坐标
            WorldCoord1(i, 2) = 1.0;
        }

        // 最小二乘解方程：A^T*A*X = A^T*B
        Eigen::MatrixXd ATA = PixelCoord1.transpose() * PixelCoord1;
        Eigen::MatrixXd ATB = PixelCoord1.transpose() * WorldCoord1;

        // LDLT分解求解（更高效且数值稳定）
        if(ATA.determinant() < 1e-10) {
            cerr << "错误: 矩阵奇异，无法求解" << endl;
            return 3; // 错误码3: 矩阵奇异
        }
        
        // 求解变换矩阵
        Eigen::MatrixXd resultMatrix = ATA.ldlt().solve(ATB);
        
        // 构建3x3变换矩阵
        matrix << resultMatrix(0, 0), resultMatrix(0, 1), resultMatrix(0, 2),
                  resultMatrix(1, 0), resultMatrix(1, 1), resultMatrix(1, 2),
                  resultMatrix(2, 0), resultMatrix(2, 1), resultMatrix(2, 2);

        // 输出矩阵信息（高精度格式）
        cout << "变换矩阵:" << endl;
        cout << fixed << setprecision(10);
        cout << matrix << endl;

        // 验证变换结果
        cout << "验证变换结果:" << endl;
        for (int i = 0; i < pointCount; ++i)
        {
            Vector3d pixel(PixelCoord[i].x(), PixelCoord[i].y(), 1.0);
            Vector3d world = matrix * pixel;

            double error_x = std::abs(world[0] - WorldCoord[i].x());
            double error_y = std::abs(world[1] - WorldCoord[i].y());
            
            cout << "像素坐标: (" << pixel[0] << ", " << pixel[1] << ") -> ";
            cout << "计算的世界坐标: (" << world[0] << ", " << world[1] << ") ";
            cout << "实际的世界坐标: (" << WorldCoord[i].x() << ", " << WorldCoord[i].y() << ")";
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

// 读取本地照片，完成角点检测和矩阵变换，输出各点像素坐标
int Algorithm(const string& imgPath, HTuple& Row, HTuple& Col)
{
    HObject  ho_Image, ho_GrayImage, ho_RegionBinary;
    HObject  ho_ImageBinary, ho_ImageSmoothed, ho_Edges, ho_LongEdges;
    HObject  ho_SmoothEdges, ho_UnionContours, ho_Rectangle;
    HObject  ho_Cross1, ho_Cross;

    HTuple  hv_Width, hv_Height, hv_Row, hv_Column;
    HTuple  hv_Phi, hv_Length1, hv_Length2, hv_PointOrder, hv_Row2;
    HTuple  hv_Col2, hv_WindowHandle, hv_i;

    if (HDevWindowStack::IsOpen())
    {
        ClearWindow(HDevWindowStack::GetActive());
    }
    if (HDevWindowStack::IsOpen())
    {
        CloseWindow(HDevWindowStack::Pop());
    }

    //读取图像
    ReadImage(&ho_Image, HTuple(imgPath.c_str()));
    Rgb1ToGray(ho_Image, &ho_GrayImage);

    //自适应二值化
    BinThreshold(ho_GrayImage, &ho_RegionBinary);

    //获取图像尺寸
    GetImageSize(ho_GrayImage, &hv_Width, &hv_Height);

    //使用region_to_bin创建黑白图像
    RegionToBin(ho_RegionBinary, &ho_ImageBinary, 0, 255, hv_Width, hv_Height);

    //对二值图像进行平滑处理（使用高斯滤波）
    GaussFilter(ho_ImageBinary, &ho_ImageSmoothed, 3);

    //使用亚像素精度的边缘检测
    EdgesSubPix(ho_ImageSmoothed, &ho_Edges, "canny", 1.5, 20, 40);

    //选择长边缘（过滤掉短小的噪声边缘）
    SelectShapeXld(ho_Edges, &ho_LongEdges, "contlength", "and", 50, 1000000);

    //对边缘进行平滑处理
    SmoothContoursXld(ho_LongEdges, &ho_SmoothEdges, 11);

    //将边缘连接成轮廓
    UnionAdjacentContoursXld(ho_SmoothEdges, &ho_UnionContours, 10, 1, "attr_keep");

    try
    {
        //拟合矩形
        FitRectangle2ContourXld(ho_UnionContours, "tukey", -1, 0, 0, 3, 2, &hv_Row, &hv_Column,
             &hv_Phi, &hv_Length1, &hv_Length2, &hv_PointOrder);

        //生成矩形轮廓并计算角点
        GenRectangle2ContourXld(&ho_Rectangle, hv_Row, hv_Column, hv_Phi, hv_Length1, hv_Length2);
        GetContourXld(ho_Rectangle, &hv_Row2, &hv_Col2);
        GenCrossContourXld(&ho_Cross1, hv_Row2, hv_Col2, 60, hv_Phi);
        hv_Row2 = hv_Row2.TupleSelectRange(0,3);
        Row = hv_Row2;
        hv_Col2 = hv_Col2.TupleSelectRange(0,3);
        Col = hv_Col2;
    }
    catch (HException& e)
    {
        qDebug() << "" << e.ErrorMessage().Text();
        return 1;
    }

    //创建窗口并显示原始图像
    SetWindowAttr("background_color","black");
    OpenWindow(0,0,hv_Width,hv_Height,0,"visible","",&hv_WindowHandle);
    HDevWindowStack::Push(hv_WindowHandle);
    if (HDevWindowStack::IsOpen())
    {
        DispObj(ho_Image, HDevWindowStack::GetActive());
    }

    //显示矩形轮廓
    if (HDevWindowStack::IsOpen())
    {
        SetColor(HDevWindowStack::GetActive(),"green");
    }
    if (HDevWindowStack::IsOpen())
    {
        SetLineWidth(HDevWindowStack::GetActive(),10);
    }
    if (HDevWindowStack::IsOpen())
    {
        DispObj(ho_Rectangle, HDevWindowStack::GetActive());
    }

    //绘制角点
    if (HDevWindowStack::IsOpen())
    {
        SetColor(HDevWindowStack::GetActive(),"red");
    }
    for (hv_i=0; hv_i<=10; hv_i+=1)
    {
        GenCrossContourXld(&ho_Cross, HTuple(hv_Row2[hv_i]), HTuple(hv_Col2[hv_i]), 200,0);
        if (HDevWindowStack::IsOpen())
        {
            DispObj(ho_Cross, HDevWindowStack::GetActive());
        }
    }

     //保存图像
     DumpWindow(hv_WindowHandle, "jpeg", "../detectedImg.jpg");

     // 保存图像后关闭窗口
     if (HDevWindowStack::IsOpen())
     {
         CloseWindow(HDevWindowStack::Pop());
     }

     return 0;
}

// OpenCV版本的圆形检测算法
int getCoords_opencv(QVector<QPointF>& WorldCoord, QVector<QPointF>& PixelCoord, double size = 75.0)
{
    // 读取图像
    cv::Mat image = cv::imread("/home/orangepi/Desktop/VisualRobot_Local/Img/capture.jpg");
    if(image.empty()) {
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
    vector<cv::Rect> searchAreas;
    vector<cv::Point2f> worldCoords;
    
    // 计算世界坐标
    double world_dis_col = 1049.0 * size / 2734.0;
    double world_dis_row = 774.0 * size / 2734.0;

    // 定义9个区域的世界坐标
    for(int row = 0; row < 3; row++) {
        for(int col = 0; col < 3; col++) {
            // 计算搜索区域
            int x1 = (col == 0) ? 0 : static_cast<int>(r + (col - 0.5) * dis_col);
            int x2 = (col == 2) ? width : static_cast<int>(r + (col + 0.5) * dis_col);
            int y1 = (row == 0) ? 0 : static_cast<int>(r + (row - 0.5) * dis_row);
            int y2 = (row == 2) ? height : static_cast<int>(r + (row + 0.5) * dis_row);
            
            searchAreas.push_back(cv::Rect(x1, y1, x2-x1, y2-y1));
            worldCoords.push_back(cv::Point2f(col * world_dis_col, row * world_dis_row));
        }
    }

    // 转换为灰度图
    cv::Mat grayImage;
    cv::cvtColor(image, grayImage, cv::COLOR_BGR2GRAY);

    // 二值化
    cv::Mat binaryImage;
    cv::threshold(grayImage, binaryImage, 200, 255, cv::THRESH_BINARY);

    // 在每个区域中检测圆
    vector<cv::Vec3f> detectedCircles;
    cv::Mat result = image.clone();

    for(size_t i = 0; i < searchAreas.size(); i++) {
        cv::Mat roi = binaryImage(searchAreas[i]);
        vector<cv::Vec3f> circles;

        // 使用Hough圆变换检测圆
        cv::HoughCircles(roi, circles, cv::HOUGH_GRADIENT, 1,
                        roi.rows/8,  // 最小圆心距离
                        40, 5,     // Canny阈值，累加器阈值
                        rmin, rmax); // 最小和最大半径

        // 找到最佳匹配的圆
        double bestFit = std::numeric_limits<double>::max();
        cv::Vec3f bestCircle;
        bool found = false;

        for(const auto& circle : circles) {
            float radius = circle[2];
            float bias = std::abs(radius - r);

            if(bias < bestFit) {
                bestFit = bias;
                bestCircle = circle;
                found = true;
            }
        }

        if(found) {
            // 调整圆心坐标到原图坐标系
            bestCircle[0] += searchAreas[i].x;
            bestCircle[1] += searchAreas[i].y;
            detectedCircles.push_back(bestCircle);

            // 存储结果
            WorldCoord.append(QPointF(worldCoords[i].x, worldCoords[i].y));
            PixelCoord.append(QPointF(bestCircle[0], bestCircle[1]));

            // 绘制结果
            cv::circle(result, cv::Point(bestCircle[0], bestCircle[1]),
                      bestCircle[2], cv::Scalar(0, 255, 0), 10);
            cv::circle(result, cv::Point(bestCircle[0], bestCircle[1]),
                      10, cv::Scalar(0, 0, 255), -1);
        }
    }

    // 保存结果图像
    cv::imwrite("/home/orangepi/Desktop/VisualRobot_Local/Img/circle_detected.jpg", result);

    return detectedCircles.empty() ? 1 : 0;
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
    cv::Mat image = cv::imread(imgPath);
    if (image.empty()) {
        qDebug() << "Error: Cannot read image file";
        return 1;
    }

    // 转换为灰度图
    cv::Mat grayImage;
    cv::cvtColor(image, grayImage, cv::COLOR_BGR2GRAY);

    // 二值化
    cv::Mat binaryImage;
    cv::threshold(grayImage, binaryImage, 128, 255, cv::THRESH_BINARY);

    // 高斯滤波平滑处理
    cv::Mat smoothedImage;
    cv::GaussianBlur(binaryImage, smoothedImage, cv::Size(3, 3), 0);

    // Canny边缘检测
    cv::Mat edges;
    cv::Canny(smoothedImage, edges, 20, 40);

    // 查找轮廓
    vector<vector<cv::Point>> contours;
    vector<cv::Vec4i> hierarchy;
    cv::findContours(edges, contours, hierarchy, 
                     cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    // 过滤短轮廓
    vector<vector<cv::Point>> longContours;
    for(const auto& contour : contours) {
        if(cv::arcLength(contour, true) > 50) {
            longContours.push_back(contour);
        }
    }

    // 寻找最大的矩形轮廓
    vector<cv::Point> rectContour;
    double maxArea = 0;
    for(const auto& contour : longContours) {
        double area = cv::contourArea(contour);
        if(area > maxArea) {
            cv::RotatedRect rect = cv::minAreaRect(contour);
            vector<cv::Point2f> boxPoints(4);
            rect.points(boxPoints.data());
            
            // 转换为整数点
            vector<cv::Point> intPoints;
            for(const auto& pt : boxPoints) {
                intPoints.push_back(cv::Point(static_cast<int>(pt.x), static_cast<int>(pt.y)));
            }
            
            maxArea = area;
            rectContour = intPoints;
        }
    }

    // 如果找到矩形
    if(!rectContour.empty()) {
        // 按照顺时针顺序排序角点
        vector<cv::Point> orderedPoints = rectContour;
        // 按y坐标排序（从上到下）
        sort(orderedPoints.begin(), orderedPoints.end(), 
             [](const cv::Point& a, const cv::Point& b) { return a.y < b.y; });
        
        // 上面的两个点按x坐标排序（从左到右）
        if(orderedPoints[0].x > orderedPoints[1].x) 
            swap(orderedPoints[0], orderedPoints[1]);
        
        // 下面的两个点按x坐标排序（从左到右）
        if(orderedPoints[2].x > orderedPoints[3].x) 
            swap(orderedPoints[2], orderedPoints[3]);

        // 存储角点坐标
        for(const auto& pt : orderedPoints) {
            Row.push_back(pt.y);
            Col.push_back(pt.x);
        }

        // 可视化结果
        cv::Mat result = image.clone();
        
        // 绘制矩形
        for(int i = 0; i < 4; i++) {
            cv::line(result, rectContour[i], rectContour[(i+1)%4], 
                    cv::Scalar(0, 255, 0), 10);
            // 绘制角点
            cv::drawMarker(result, cv::Point(Col[i], Row[i]),
                          cv::Scalar(0, 0, 255), cv::MARKER_CROSS, 200, 10);
        }

        // 保存结果图像
        cv::imwrite("../detectedImg.jpg", result);
        
        return 0;
    }

    return 1;
}

int getCoords(QVector<QPointF>& WorldCoord, QVector<QPointF>& PixelCoord, double size = 75.0)
{
    // Local iconic variables
    HObject  ho_Image, ho_Edges, ho_ContoursSplit;
    HObject  ho_SelectedContours, ho_UnionContours, ho_MatchedContours;
    HObject  ho_SingleContour, ho_ColorImage, ho_Contour, ho_Circle;

    // Local control variables
    HTuple  hv_Width, hv_Height, hv_r, hv_dis_col;
    HTuple  hv_dis_row, hv_rmin, hv_rmax, hv_area_rows_min;
    HTuple  hv_area_rows_max, hv_area_cols_min, hv_area_cols_max;
    HTuple  hv_world_coords_x, hv_world_coords_y, hv_Number;
    HTuple  hv_pixel_coords_x, hv_pixel_coords_y, hv_valid_world_coords_x;
    HTuple  hv_valid_world_coords_y, hv_area_idx, hv_min_row;
    HTuple  hv_max_row, hv_min_col, hv_max_col, hv_best_circle_found;
    HTuple  hv_best_row, hv_best_col, hv_best_radius, hv_best_bias;
    HTuple  hv_i, hv_Row, hv_Column, hv_Radius, hv_StartPhi;
    HTuple  hv_EndPhi, hv_PointOrder, hv_bias, hv_Exception;
    HTuple  hv_WindowHandle, hv_NumMatched;

    if (HDevWindowStack::IsOpen())
    {
        ClearWindow(HDevWindowStack::GetActive());
    }
    if (HDevWindowStack::IsOpen())
    {
        CloseWindow(HDevWindowStack::Pop());
    }

    //读取图像
    ReadImage(&ho_Image, "/home/orangepi/Desktop/VisualRobot_local/Img/capture.jpg");

    //获取图像尺寸
    GetImageSize(ho_Image, &hv_Width, &hv_Height);

    //计算参数（基于伪代码中的比例）
    hv_r = (hv_Height*316.0)/2182.0;
    hv_dis_col = (hv_Width*1049.0)/2734.0;
    hv_dis_row = (hv_Height*774.0)/2182.0;
    hv_rmin = hv_r-10;
    hv_rmax = hv_r+10;

    //分离区域定义 - 行范围
    hv_area_rows_min.Clear();
    hv_area_rows_min[0] = 0;
    hv_area_rows_min[1] = 0;
    hv_area_rows_min[2] = 0;
    hv_area_rows_min.Append(hv_r+(hv_dis_row/2));
    hv_area_rows_min.Append(hv_r+(hv_dis_row/2));
    hv_area_rows_min.Append(hv_r+(hv_dis_row/2));
    hv_area_rows_min.Append(hv_r+((3*hv_dis_row)/2));
    hv_area_rows_min.Append(hv_r+((3*hv_dis_row)/2));
    hv_area_rows_min.Append(hv_r+((3*hv_dis_row)/2));
    hv_area_rows_max.Clear();
    hv_area_rows_max.Append(hv_r+(hv_dis_row/2));
    hv_area_rows_max.Append(hv_r+(hv_dis_row/2));
    hv_area_rows_max.Append(hv_r+(hv_dis_row/2));
    hv_area_rows_max.Append(hv_r+((3*hv_dis_row)/2));
    hv_area_rows_max.Append(hv_r+((3*hv_dis_row)/2));
    hv_area_rows_max.Append(hv_r+((3*hv_dis_row)/2));
    hv_area_rows_max.Append(hv_Height-1);
    hv_area_rows_max.Append(hv_Height-1);
    hv_area_rows_max.Append(hv_Height-1);

    //分离区域定义 - 列范围
    hv_area_cols_min.Clear();
    hv_area_cols_min[0] = 0;
    hv_area_cols_min.Append(hv_r+(hv_dis_col/2));
    hv_area_cols_min.Append(hv_r+((3*hv_dis_col)/2));
    hv_area_cols_min.Append(0);
    hv_area_cols_min.Append(hv_r+(hv_dis_col/2));
    hv_area_cols_min.Append(hv_r+((3*hv_dis_col)/2));
    hv_area_cols_min.Append(0);
    hv_area_cols_min.Append(hv_r+(hv_dis_col/2));
    hv_area_cols_min.Append(hv_r+((3*hv_dis_col)/2));
    hv_area_cols_max.Clear();
    hv_area_cols_max.Append(hv_r+(hv_dis_col/2));
    hv_area_cols_max.Append(hv_r+((3*hv_dis_col)/2));
    hv_area_cols_max.Append(hv_Width-1);
    hv_area_cols_max.Append(hv_r+(hv_dis_col/2));
    hv_area_cols_max.Append(hv_r+((3*hv_dis_col)/2));
    hv_area_cols_max.Append(hv_Width-1);
    hv_area_cols_max.Append(hv_r+(hv_dis_col/2));
    hv_area_cols_max.Append(hv_r+((3*hv_dis_col)/2));
    hv_area_cols_max.Append(hv_Width-1);

    //分离世界坐标定义
    double world_dis_col = 1049.0 * size / 2734.0;
    double world_dis_row = 774.0 * size / 2734.0;
    hv_world_coords_x.Clear();
    hv_world_coords_x[0] = 0.0;
    hv_world_coords_x[1] = world_dis_col;
    hv_world_coords_x[2] = 2.0 * world_dis_col;
    hv_world_coords_x[3] = 0.0;
    hv_world_coords_x[4] = world_dis_col;
    hv_world_coords_x[5] = 2.0 * world_dis_col;
    hv_world_coords_x[6] = 0.0;
    hv_world_coords_x[7] = world_dis_col;
    hv_world_coords_x[8] = 2.0 * world_dis_col;
    hv_world_coords_y.Clear();
    hv_world_coords_y[0] = 0.0;
    hv_world_coords_y[1] = 0.0;
    hv_world_coords_y[2] = 0.0;
    hv_world_coords_y[3] = world_dis_row;
    hv_world_coords_y[4] = world_dis_row;
    hv_world_coords_y[5] = world_dis_row;
    hv_world_coords_y[6] = 2.0 * world_dis_row;
    hv_world_coords_y[7] = 2.0 * world_dis_row;
    hv_world_coords_y[8] = 2.0 * world_dis_row;

    //亚像素边缘检测
    EdgesSubPix(ho_Image, &ho_Edges, "canny", 2, 15, 40);

    //分割边缘为独立线段/圆弧
    SegmentContoursXld(ho_Edges, &ho_ContoursSplit, "lines_circles", 5, 4, 2);

    //选择轮廓
    SelectContoursXld(ho_ContoursSplit, &ho_SelectedContours, "contour_length", 100,
        5000, -0.5, 0.5);

    //合并共圆轮廓
    UnionCocircularContoursXld(ho_SelectedContours, &ho_UnionContours, 0.1, 0.1, 0.5,
        20, 10, 30, "true", 1);

    //拟合圆形并获取参数
    CountObj(ho_UnionContours, &hv_Number);
    hv_pixel_coords_x = HTuple();
    hv_pixel_coords_y = HTuple();
    hv_valid_world_coords_x = HTuple();
    hv_valid_world_coords_y = HTuple();

    //创建空对象用于存储匹配的轮廓
    GenEmptyObj(&ho_MatchedContours);

    for (hv_area_idx=0; hv_area_idx<=8; hv_area_idx+=1)
    {
      //获取当前区域范围
      hv_min_row = (HTuple(0).TupleConcat(HTuple(hv_area_rows_min[hv_area_idx]))).TupleMax();
      hv_max_row = ((hv_Height-1).TupleConcat(HTuple(hv_area_rows_max[hv_area_idx]))).TupleMin();
      hv_min_col = (HTuple(0).TupleConcat(HTuple(hv_area_cols_min[hv_area_idx]))).TupleMax();
      hv_max_col = ((hv_Width-1).TupleConcat(HTuple(hv_area_cols_max[hv_area_idx]))).TupleMin();

      hv_best_circle_found = 0;
      hv_best_row = 0.0;
      hv_best_col = 0.0;
      hv_best_radius = 0.0;
      hv_best_bias = 0.0;

      //遍历所有检测到的圆
      {
      HTuple end_val64 = hv_Number;
      HTuple step_val64 = 1;
      for (hv_i=1; hv_i.Continue(end_val64, step_val64); hv_i += step_val64)
      {
        SelectObj(ho_UnionContours, &ho_SingleContour, hv_i);

        //添加错误处理
        try
        {
          FitCircleContourXld(ho_SingleContour, "geotukey", -1, 0, 0, 3, 2, &hv_Row,
              &hv_Column, &hv_Radius, &hv_StartPhi, &hv_EndPhi, &hv_PointOrder);

          //检查圆是否在当前区域内且半径符合要求
          if (0 != (HTuple(HTuple(HTuple(HTuple(HTuple(int(hv_Row>=hv_min_row)).TupleAnd(int(hv_Row<=hv_max_row))).TupleAnd(int(hv_Column>=hv_min_col))).TupleAnd(int(hv_Column<=hv_max_col))).TupleAnd(int(hv_Radius>=hv_rmin))).TupleAnd(int(hv_Radius<=hv_rmax))))
          {
            //计算与期望半径的偏差
            hv_bias = (hv_Radius-hv_r).TupleAbs();

            //选择最接近期望半径的圆
            if (0 != (HTuple(hv_best_circle_found.TupleNot()).TupleOr(int(hv_bias<hv_best_bias))))
            {
              hv_best_circle_found = 1;
              hv_best_row = hv_Row;
              hv_best_col = hv_Column;
              hv_best_radius = hv_Radius;
              hv_best_bias = hv_bias;
            }
          }
        }
        // catch (Exception)
        catch (HException &HDevExpDefaultException)
        {
          HDevExpDefaultException.ToHTuple(&hv_Exception);
          //跳过无法处理的轮廓
          continue;
        }
      }
      }

      //如果找到符合条件的圆，添加到结果列表
      if (0 != hv_best_circle_found)
      {
        hv_pixel_coords_x = hv_pixel_coords_x.TupleConcat(hv_best_row);
        hv_pixel_coords_y = hv_pixel_coords_y.TupleConcat(hv_best_col);
        hv_valid_world_coords_x = hv_valid_world_coords_x.TupleConcat(HTuple(hv_world_coords_x[hv_area_idx]));
        hv_valid_world_coords_y = hv_valid_world_coords_y.TupleConcat(HTuple(hv_world_coords_y[hv_area_idx]));

        //重新选择最佳轮廓并添加到匹配轮廓集合
        {
        HTuple end_val99 = hv_Number;
        HTuple step_val99 = 1;
        for (hv_i=1; hv_i.Continue(end_val99, step_val99); hv_i += step_val99)
        {
          SelectObj(ho_UnionContours, &ho_SingleContour, hv_i);
          try
          {
            FitCircleContourXld(ho_SingleContour, "geotukey", -1, 0, 0, 3, 2, &hv_Row,
                &hv_Column, &hv_Radius, &hv_StartPhi, &hv_EndPhi, &hv_PointOrder);
            if (0 != (HTuple(HTuple(int(((hv_Row-hv_best_row).TupleAbs())<1)).TupleAnd(int(((hv_Column-hv_best_col).TupleAbs())<1))).TupleAnd(int(((hv_Radius-hv_best_radius).TupleAbs())<1))))
            {
              ConcatObj(ho_MatchedContours, ho_SingleContour, &ho_MatchedContours);
              break;
            }
          }
          // catch (Exception)
          catch (HException &HDevExpDefaultException)
          {
            HDevExpDefaultException.ToHTuple(&hv_Exception);
            continue;
          }
        }
        }
      }
    }

    //创建结果图像
    Compose3(ho_Image, ho_Image, ho_Image, &ho_ColorImage);

    //打开一个临时窗口用于绘制
    SetWindowAttr("background_color","black");
    OpenWindow(0,0,hv_Width,hv_Height,0,"visible","",&hv_WindowHandle);
    HDevWindowStack::Push(hv_WindowHandle);
    if (HDevWindowStack::IsOpen())
      DispObj(ho_ColorImage, HDevWindowStack::GetActive());

    //绘制所有检测到的圆轮廓（绿色）
    CountObj(ho_MatchedContours, &hv_NumMatched);
    if (0 != (int(hv_NumMatched>0)))
    {
      if (HDevWindowStack::IsOpen())
        SetColor(HDevWindowStack::GetActive(),"green");
      if (HDevWindowStack::IsOpen())
        SetLineWidth(HDevWindowStack::GetActive(),10);
      {
      HTuple end_val126 = hv_NumMatched;
      HTuple step_val126 = 1;
      for (hv_i=1; hv_i.Continue(end_val126, step_val126); hv_i += step_val126)
      {
        SelectObj(ho_MatchedContours, &ho_Contour, hv_i);
        if (HDevWindowStack::IsOpen())
          DispObj(ho_Contour, HDevWindowStack::GetActive());
      }
      }
    }

    //绘制所有圆心（红色）
    if (0 != (int((hv_pixel_coords_x.TupleLength())>0)))
    {
      if (HDevWindowStack::IsOpen())
        SetColor(HDevWindowStack::GetActive(),"red");
      if (HDevWindowStack::IsOpen())
        SetDraw(HDevWindowStack::GetActive(),"fill");
      {
      HTuple end_val136 = (hv_pixel_coords_x.TupleLength())-1;
      HTuple step_val136 = 1;
      for (hv_i=0; hv_i.Continue(end_val136, step_val136); hv_i += step_val136)
      {
        GenCircle(&ho_Circle, HTuple(hv_pixel_coords_x[hv_i]), HTuple(hv_pixel_coords_y[hv_i]), 10);
        if (HDevWindowStack::IsOpen())
          DispObj(ho_Circle, HDevWindowStack::GetActive());
      }
      }
    }

    //等待一小段时间确保绘制完成
    WaitSeconds(0.5);

    //将窗口内容保存为图像
    DumpWindow(hv_WindowHandle, "jpg", "/home/orangepi/Desktop/VisualRobot_local/Img/circle_detected.jpg");

    //关闭窗口
    if (HDevWindowStack::IsOpen())
      CloseWindow(HDevWindowStack::Pop());

    // 将结果存入输出向量
    for (int i = 0; i < hv_valid_world_coords_x.TupleLength(); i++)
    {
        // 世界坐标 (x, y)
        WorldCoord.append(QPointF(
            hv_valid_world_coords_x[i].D(),
            hv_valid_world_coords_y[i].D()
        ));

        // 像素坐标 (列, 行) -> 注意Halcon中Row是y坐标，Column是x坐标
        PixelCoord.append(QPointF(
            hv_pixel_coords_y[i].D(),  // Column -> x
            hv_pixel_coords_x[i].D()   // Row -> y
        ));
    }

    return 0;
}
