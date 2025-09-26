#include "RegionDetector.h"
#include <opencv2/imgproc.hpp>
#include <QDebug>

RegionDetector::RegionDetector(QObject *parent) : QObject(parent)
{
    m_threshold = 127;
    m_minArea = 100;
    m_maxArea = 1000000; // 默认不设上限
}

QVector<QRect> RegionDetector::detectRegions(const cv::Mat &inputImage)
{
    QVector<QRect> regions;
    
    if (inputImage.empty()) {
        qDebug() << "RegionDetector: 输入图像为空";
        return regions;
    }

    cv::Mat grayImage, binaryImage;
    
    // 转换为灰度图
    if (inputImage.channels() == 3) {
        cv::cvtColor(inputImage, grayImage, cv::COLOR_BGR2GRAY);
    } else {
        grayImage = inputImage.clone();
    }
    
    // 二值化处理
    cv::threshold(grayImage, binaryImage, m_threshold, 255, cv::THRESH_BINARY);
    
    // 查找轮廓
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(binaryImage, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    
    // 处理每个轮廓
    for (const auto &contour : contours) {
        double area = cv::contourArea(contour);
        
        // 过滤面积过小或过大的轮廓
        if (area < m_minArea || area > m_maxArea) {
            continue;
        }
        
        // 获取轮廓的边界矩形
        cv::Rect rect = cv::boundingRect(contour);
        regions.append(QRect(rect.x, rect.y, rect.width, rect.height));
    }
    
    return regions;
}

QVector<QRect> RegionDetector::detectRegionsWithBias(const cv::Mat &inputImage, int bias)
{
    QVector<QRect> regions;
    
    if (inputImage.empty()) {
        qDebug() << "RegionDetector: 输入图像为空";
        return regions;
    }

    cv::Mat grayImage, binaryImage;
    
    // 转换为灰度图
    if (inputImage.channels() == 3) {
        cv::cvtColor(inputImage, grayImage, cv::COLOR_BGR2GRAY);
    } else {
        grayImage = inputImage.clone();
    }
    
    // 二值化处理
    cv::threshold(grayImage, binaryImage, m_threshold, 255, cv::THRESH_BINARY);
    
    // 查找轮廓
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(binaryImage, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    
    // 处理每个轮廓
    for (const auto &contour : contours) {
        double area = cv::contourArea(contour);
        
        // 过滤面积过小或过大的轮廓
        if (area < m_minArea || area > m_maxArea) {
            continue;
        }
        
        // 获取轮廓的边界矩形
        cv::Rect rect = cv::boundingRect(contour);
        
        // 扩展矩形，上下左右各扩展bias像素
        QRect expandedRect = expandRect(rect, bias, inputImage.size());
        regions.append(expandedRect);
    }
    
    return regions;
}

void RegionDetector::setThreshold(int threshold)
{
    m_threshold = threshold;
}

void RegionDetector::setMinArea(int minArea)
{
    m_minArea = minArea;
}

void RegionDetector::setMaxArea(int maxArea)
{
    m_maxArea = maxArea;
}

QRect RegionDetector::expandRect(const cv::Rect &rect, int bias, const cv::Size &imageSize)
{
    int x1 = std::max(0, rect.x - bias);
    int y1 = std::max(0, rect.y - bias);
    int x2 = std::min(imageSize.width - 1, rect.x + rect.width + bias);
    int y2 = std::min(imageSize.height - 1, rect.y + rect.height + bias);
    
    return QRect(x1, y1, x2 - x1, y2 - y1);
}
