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
#include <fstream>
#include <nlohmann/json.hpp>

#define ERROR 2
#define WARNNING 1
#define INFO 0

using namespace Eigen;
using namespace std;
using namespace cv;
using json = nlohmann::json;

//读取json
bool LoadYoloDetections(const std::string& jsonPath, std::vector<YoloDetBox>& dets)
{
    dets.clear();

    std::ifstream ifs(jsonPath);
    if (!ifs.is_open()) 
    {
        std::cerr << "Failed to open detection json: " << jsonPath << std::endl;
        return false;
    }

    json j;
    try 
    {
        ifs >> j;
    } 
    catch (const std::exception& e) 
    {
        std::cerr << "Json parse error: " << e.what() << std::endl;
        return false;
    }

    if (!j.is_array()) 
    {
        std::cerr << "Json format error: root is not array.\n";
        return false;
    }

    for (auto& item : j) 
    {
        if (!item.contains("bbox")) 
        {
            continue;
        }
        auto jb = item["bbox"];
        if (!jb.is_array() || jb.size() != 4) 
        {
            continue;
        }

        float xmin = jb[0].get<float>();
        float ymin = jb[1].get<float>();
        float xmax = jb[2].get<float>();
        float ymax = jb[3].get<float>();

        YoloDetBox box;
        box.bbox = cv::Rect2f(cv::Point2f(xmin, ymin), cv::Point2f(xmax, ymax));

        if (item.contains("class"))
        {
            box.cls = item["class"].get<std::string>();
        }
        else
        {
            box.cls = "";
        }

        if (item.contains("class_id"))
        {
            box.class_id = item["class_id"].get<int>();
        }

        if (item.contains("confidence"))
        {
            box.confidence = item["confidence"].get<float>();
        }

        // ★★ 这里就只要 defect1 / defect2 ★★
        if (box.cls != "defect1" && box.cls != "defect2") 
        {
            continue;
        }

        dets.push_back(std::move(box));
    }

    return true;
}

//放大
cv::Rect EnlargeBBox(const cv::Rect2f& box, int imgW, int imgH, float scale)
{
    float xmin = box.x;
    float ymin = box.y;
    float xmax = box.x + box.width;
    float ymax = box.y + box.height;

    float cx = 0.5f * (xmin + xmax);
    float cy = 0.5f * (ymin + ymax);
    float w  = (xmax - xmin);
    float h  = (ymax - ymin);

    float newW = w * scale;
    float newH = h * scale;

    float newXmin = cx - newW * 0.5f;
    float newYmin = cy - newH * 0.5f;
    float newXmax = cx + newW * 0.5f;
    float newYmax = cy + newH * 0.5f;

    // 限制在图像范围内
    newXmin = std::max(0.0f, newXmin);
    newYmin = std::max(0.0f, newYmin);
    newXmax = std::min((float)imgW - 1, newXmax);
    newYmax = std::min((float)imgH - 1, newYmax);

    if (newXmax <= newXmin || newYmax <= newYmin) 
    {
        return cv::Rect(); // 空
    }

    cv::Rect r(
        (int)std::floor(newXmin),
        (int)std::floor(newYmin),
        (int)std::ceil(newXmax - newXmin),
        (int)std::ceil(newYmax - newYmin)
    );

    r &= cv::Rect(0, 0, imgW, imgH);
    return r;
}

//涂黑
cv::Mat MaskImageByYoloJson(const cv::Mat& input, const std::string& jsonPath, float scale)
{
    CV_Assert(!input.empty());

    int imgW = input.cols;
    int imgH = input.rows;

    std::vector<YoloDetBox> dets;
    if (!LoadYoloDetections(jsonPath, dets)) 
    {
        // 失败就返回一张全黑，避免程序崩
        return cv::Mat::zeros(input.size(), input.type());
    }

    // 单通道 mask
    cv::Mat mask = cv::Mat::zeros(input.size(), CV_8UC1);

    for (const auto& d : dets) 
    {
        cv::Rect enlarged = EnlargeBBox(d.bbox, imgW, imgH, scale);
        if (enlarged.area() <= 0) continue;

        cv::rectangle(mask, enlarged, cv::Scalar(255), cv::FILLED);
    }

    cv::Mat result = cv::Mat::zeros(input.size(), input.type());
    input.copyTo(result, mask);  // 只有 mask=255 的地方会拷贝原图
    return result;
}

/**
 * @brief 创建目录的辅助函数
 * @param path 要创建的目录路径
 * @return 创建成功返回true，失败返回false
 * 
 * 使用Qt的QDir类创建指定路径的目录，包括所有必要的父目录。
 * 如果目录已存在，也会返回true。
 */
bool CreateDirectory(const string& path)
{
    // 变量定义
    QDir dir(QString::fromStdString(path)); // QDir对象用于目录操作
    
    return dir.mkpath("."); // 创建路径及其所有父目录
}

/**
 * @brief 使用OpenCV检测标定板上的圆形标记并获取坐标
 * @param WorldCoord 输出参数，存储检测到的世界坐标
 * @param PixelCoord 输出参数，存储检测到的像素坐标
 * @param size 标定板尺寸，默认为100.0
 * @return 成功返回0，失败返回1
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
 */
int GetCoordsOpenCV(QVector<QPointF>& WorldCoord, QVector<QPointF>& PixelCoord, double size = 100.0)
{
    // 变量定义
    Mat image;                              // 输入图像
    int height, width;                      // 图像高度和宽度
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
    r = (height * 316.0) / 2182.0;          // 预期圆半径
    dis_col = (width * 1049.0) / 2734.0;    // 列方向圆心间距
    dis_row = (height * 774.0) / 2182.0;    // 行方向圆心间距
    rmin = r - 10;                         // 最小圆半径
    rmax = r + 10;                         // 最大圆半径
    
    // 计算世界坐标间距
    world_dis_col = 1049.0 * size / 2734.0;  // 世界坐标列间距
    world_dis_row = 774.0 * size / 2734.0;   // 世界坐标行间距

    // 定义9个区域的世界坐标（3x3网格）
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

    // 二值化处理
    threshold(grayImage, binaryImage, 200, 255, THRESH_BINARY);

    // 在每个区域中检测圆
    result = image.clone();

    for(size_t i = 0; i < searchAreas.size(); i++) 
    {
        // 获取当前搜索区域的ROI
        Mat roi = binaryImage(searchAreas[i]);
        vector<Vec3f> circles;

        // 使用Hough圆变换检测圆
        HoughCircles(roi, circles, HOUGH_GRADIENT, 1,
                        roi.rows/8,  // 最小圆心距离
                        40, 5,       // Canny阈值，累加器阈值
                        rmin, rmax); // 最小和最大半径

        // 找到最佳匹配的圆（与预期半径最接近的圆）
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
        // 保存世界坐标和像素坐标
        WorldCoord.append(QPointF(detectedWorldCoords[i].x, detectedWorldCoords[i].y));
        PixelCoord.append(QPointF(detectedCenters[i].x, detectedCenters[i].y));

        // 绘制结果：绿色圆圈表示圆的边界，红色圆点表示圆心
        circle(result, detectedCenters[i], detectedRadii[i], Scalar(0, 255, 0), 10);
        circle(result, detectedCenters[i], 10, Scalar(0, 0, 255), -1);
    }

    // 保存结果图像
    imwrite("../Img/circle_detected.jpg", result);

    // 如果没有检测到圆，返回失败；否则返回成功
    return detectedCenters.empty() ? 1 : 0;
}

/**
 * @brief 从文件读取变换矩阵
 * @param filename 矩阵文件路径
 * @return 读取到的3x3变换矩阵
 * @throws runtime_error 如果文件打开失败或解析错误
 * 
 * 从指定文件中读取3x3变换矩阵，文件格式应为3行3列的浮点数矩阵。
 * 每行的元素之间用空格分隔。
 */
Matrix3d ReadTransformationMatrix(const string& filename)
{
    // 变量定义
    ifstream file;     // 文件流对象
    Matrix3d matrix;   // 变换矩阵
    string line;       // 文件行内容
    istringstream iss; // 字符串流用于解析
    
    // 打开文件
    file.open(filename);
    if (!file)
    {
        throw runtime_error("Cannot open file: " + filename);
    }

    // 逐行读取文件内容，解析3x3矩阵
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

/**
 * @brief 使用OpenCV检测图像中的矩形
 * @param imgPath 输入图像路径
 * @param Row 输出参数，存储矩形角点的行坐标
 * @param Col 输出参数，存储矩形角点的列坐标
 * @return 成功返回0，失败返回1
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
 * 检测到的矩形角点按照顺时针顺序存储，顺序为：左上角、右上角、左下角、右下角。
 */
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

    // 二值化处理
    threshold(grayImage, binaryImage, 128, 255, THRESH_BINARY);

    // 高斯滤波平滑处理，减少噪声
    GaussianBlur(binaryImage, smoothedImage, Size(3, 3), 0);

    // Canny边缘检测
    Canny(smoothedImage, edges, 20, 40);

    // 查找轮廓
    findContours(edges, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    // 过滤短轮廓，只保留周长大于50的轮廓
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
            // 计算最小外接矩形
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

        // 亚像素精度优化角点坐标
        if (!rectContour.empty()) 
        {
            // 将角点转换为Point2f格式用于亚像素优化
            corners.clear();
            for (int i = 0; i < 4; i++) 
            {
                corners.push_back(Point2f(Col[i], Row[i]));
            }
            
            // 配置亚像素优化参数
            winSize = Size(5, 5);                  // 搜索窗口大小
            zeroZone = Size(-1, -1);               // 死区大小，(-1,-1)表示没有死区
            criteria = TermCriteria(TermCriteria::EPS + TermCriteria::MAX_ITER, 40, 0.001); // 终止条件
            
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

        // 绘制矩形轮廓
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

/**
 * @brief 使用PCA主轴方向计算点集的有向包围盒(OBB)，用于替代minAreaRect以获得更符合“主轴/最长方向”的旋转框
 * @param pts 输入点集（通常是某个连通域/轮廓的像素点）
 * @return PCA主轴对齐的旋转矩形
 */
RotatedRect GetOBBByPCA(const vector<Point>& pts)
{
    // 基本保护
    if (pts.size() < 10) 
    {
        // 点太少时退化为 minAreaRect，避免崩
        return minAreaRect(pts);
    }

    // 1) PCA 输入矩阵：N x 2
    Mat data((int)pts.size(), 2, CV_64F);
    for (int i = 0; i < (int)pts.size(); ++i) 
    {
        data.at<double>(i, 0) = pts[i].x;
        data.at<double>(i, 1) = pts[i].y;
    }

    // 2) PCA：取第一主轴
    PCA pca(data, Mat(), PCA::DATA_AS_ROW);
    Point2d mean(pca.mean.at<double>(0, 0), pca.mean.at<double>(0, 1));

    Vec2d v(pca.eigenvectors.at<double>(0, 0), pca.eigenvectors.at<double>(0, 1));
    double theta = std::atan2(v[1], v[0]); // rad

    // 3) 旋转到主轴坐标系并求包围盒 min/max
    double ca = std::cos(-theta), sa = std::sin(-theta);
    double minx =  1e18, maxx = -1e18, miny =  1e18, maxy = -1e18;

    for (const auto& p : pts) 
    {
        double x = p.x - mean.x;
        double y = p.y - mean.y;
        double xr = ca * x - sa * y;
        double yr = sa * x + ca * y;

        minx = std::min(minx, xr); maxx = std::max(maxx, xr);
        miny = std::min(miny, yr); maxy = std::max(maxy, yr);
    }

    // 4) 主轴坐标系下中心和尺寸
    Point2d cR((minx + maxx) * 0.5, (miny + maxy) * 0.5);
    Size2d  sz(maxx - minx, maxy - miny);

    // 5) 中心旋回原坐标系
    double ca2 = std::cos(theta), sa2 = std::sin(theta);
    Point2d cW(
        ca2 * cR.x - sa2 * cR.y + mean.x,
        sa2 * cR.x + ca2 * cR.y + mean.y
    );

    // 6) 输出 RotatedRect（角度用度）
    float angleDeg = (float)(theta * 180.0 / CV_PI);
    return RotatedRect(Point2f((float)cW.x, (float)cW.y), Size2f((float)sz.width, (float)sz.height), angleDeg);
}

/**
 * @brief 处理图像并计算单个物体的尺寸
 * @param input 输入图像
 * @param params 处理参数，包含阈值、模糊核大小等
 * @param bias 比例偏差，用于将像素单位转换为实际单位
 * @return 包含宽度、高度、角度和处理后图像的结果结构体
 * 
 * 该函数实现了单个物体的尺寸测量功能，主要步骤包括：
 * 1. 图像预处理（灰度化、滤波、二值化、形态学操作）
 * 2. 轮廓提取和过滤
 * 3. 寻找最大面积的轮廓
 * 4. 计算最小外接矩形
 * 5. 计算物体尺寸和角度
 * 6. 绘制结果并返回
 */
Result CalculateLength(const cv::Mat& input, const Params& params, double bias)
{
    Result result;
    if (input.empty()) 
    { 
        /* ... */ 
    }

    cv::Mat croppedInput;
    {
        const std::string jsonPath = "/home/orangepi/Desktop/VisualRobot_Local/Img/capture_detections.json";

        float enlargeScale = 1.0f; // 按需调整

        croppedInput = MaskImageByYoloJson(input, jsonPath, enlargeScale);

        // 如果 JSON 不存在或无 defect1/2，则 fallback 使用原图
        if (croppedInput.empty())
        {
            croppedInput = input.clone();
        }
    }

    cv::Mat gray;
    if (croppedInput.channels() == 3)
    {
        cv::cvtColor(croppedInput, gray, cv::COLOR_BGR2GRAY);
    }
    else
    {
        gray = croppedInput.clone();
    }

    // 1) 轻微降噪
    cv::GaussianBlur(gray, gray, cv::Size(5,5), 0);

    // 2) OTSU 阈值
    cv::Mat binary;
    cv::threshold(gray, binary, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);

    int imgW = gray.cols;
    int imgH = gray.rows;

    cv::Mat binaryMerged;
    // 半径 r 可以根据图像分辨率来设，也可以放到 Params 里配置
    int mergeRadius = (params.mergeRadius > 0) ? params.mergeRadius : std::max(imgW, imgH) / 150;   // 大概 5~15 像素

    mergeRadius = std::max(1, mergeRadius);
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(2*mergeRadius+1, 2*mergeRadius+1));

    // 闭运算 = 膨胀 + 腐蚀，更能填补物体内部的小空洞
    cv::morphologyEx(binary, binaryMerged, cv::MORPH_CLOSE, kernel, cv::Point(-1,-1), 1);

    // 3) 在 binaryMerged 上做连通域
    cv::Mat labels, stats, centroids;
    int n = cv::connectedComponentsWithStats(binaryMerged, labels, stats, centroids, 8, CV_32S);

    // 4) 过滤太小的连通域
    struct Obj {
        cv::RotatedRect rr;
        int area;
    };
    std::vector<Obj> objects;

    int objAreaMin = (params.areaMin > 0) ? params.areaMin : 5000;

    for (int i = 1; i < n; ++i) 
    {  // 0 是背景
        int area = stats.at<int>(i, cv::CC_STAT_AREA);
        if (area < objAreaMin) 
        {
            continue;
        }

        int x = stats.at<int>(i, cv::CC_STAT_LEFT);
        int y = stats.at<int>(i, cv::CC_STAT_TOP);
        int w = stats.at<int>(i, cv::CC_STAT_WIDTH);
        int h = stats.at<int>(i, cv::CC_STAT_HEIGHT);

        std::vector<cv::Point> pts;

        // ⚠ 这里用的是“binaryMerged 的 label + 原始 binary 作为掩码”
        for (int yy = y; yy < y + h; ++yy) 
        {
            const int*   rowL = labels.ptr<int>(yy);
            const uchar* rowB = binary.ptr<uchar>(yy);  // 原始二值
            for (int xx = x; xx < x + w; ++xx) 
            {
                if (rowL[xx] == i && rowB[xx] > 0) 
                {
                    pts.emplace_back(xx, yy);
                }
            }
        }

        if (pts.size() < 80) 
        {
            continue;
        }

        cv::RotatedRect rr = GetOBBByPCA(pts);
        objects.push_back({ rr, area });
    }

    // ========= 后面画框、算长宽角度=========
    cv::Mat color;
    cv::cvtColor(gray, color, cv::COLOR_GRAY2BGR);
    int thickness = std::max(2, (int)std::round(imgW * 0.002));

    if (objects.empty()) 
    {
        std::cerr << "没有满足面积要求的物体\n";
        result.widths.push_back(0);
        result.heights.push_back(0);
        result.angles.push_back(0);
        result.image = color;
        return result;
    }

    // 按 x 排序
    std::sort(objects.begin(), objects.end(),
              [](const Obj& a, const Obj& b){
                  return a.rr.center.x < b.rr.center.x;
              });

    int idx = 1;
    for (const auto& obj : objects) 
    {
        cv::RotatedRect rr = obj.rr;

        cv::Point2f v[4];
        rr.points(v);
        for (int k = 0; k < 4; ++k)
        {
            cv::line(color, v[k], v[(k+1)%4], cv::Scalar(0,255,0), thickness);
        }

        float L = std::max(rr.size.width,  rr.size.height);
        float S = std::min(rr.size.width,  rr.size.height);
        float angle = rr.angle;
        if (rr.size.width < rr.size.height) 
        {
            angle += 90.0f;
        }

        result.heights.push_back(L * bias);
        result.widths.push_back(S * bias);
        result.angles.push_back(angle);

        // 编号
        std::string label = std::to_string(idx++);
        double fontScale = std::max(1.6, imgW / 700.0);
        int fontFace  = cv::FONT_HERSHEY_SIMPLEX;
        int fontThick = std::max(2, thickness + 1);
        int base = 0;
        cv::Size ts = cv::getTextSize(label, fontFace, fontScale, fontThick, &base);
        cv::Point c((int)rr.center.x, (int)rr.center.y);
        cv::Point org(c.x - ts.width/2, c.y + ts.height/2);

        cv::Rect bg(org.x - 8, org.y - ts.height - 8, ts.width + 16, ts.height + 16);
        bg &= cv::Rect(0,0,color.cols,color.rows);
        cv::rectangle(color, bg, cv::Scalar(255,255,255), cv::FILLED);
        cv::rectangle(color, bg, cv::Scalar(0,0,0), 1);
        cv::putText(color, label, org, fontFace, fontScale, cv::Scalar(0,255,0), fontThick, cv::LINE_AA);
    }

    result.image = color;
    return result;
}

/**
 * @brief 基于连通域的多目标检测与测量函数
 * @param input 输入图像
 * @param params 处理参数，包含阈值、模糊核大小、面积最小值等
 * @param bias 比例偏差，用于将像素单位转换为实际物理单位
 * @return 包含所有检测目标的宽度、高度、角度和结果图像的结构体
 * 
 * 该函数实现了基于连通域的多目标检测与测量算法，主要步骤包括：
 * 1. 图像预处理（灰度化、滤波、二值化、形态学操作）
 * 2. 轮廓提取和面积过滤
 * 3. 第一阶段：处理所有轮廓，收集信息并执行Canny边缘检测
 * 4. 边缘处理：排除与轮廓近似的边缘，对剩余边缘进行平滑拟合
 * 5. 第二阶段：绘制轮廓边界框和序号，根据边缘情况决定颜色
 * 6. 计算每个目标的尺寸和角度，并保存结果
 * 
 * 检测到的目标将按照从1开始的序号进行标记，边界框颜色根据是否存在非轮廓边缘决定：
 * - 红色：存在非轮廓边缘
 * - 绿色：无非轮廓边缘
 */
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
