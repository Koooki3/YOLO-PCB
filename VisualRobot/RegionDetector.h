#ifndef REGIONDETECTOR_H
#define REGIONDETECTOR_H

#include <QObject>
#include <QVector>
#include <QRect>
#include <opencv2/opencv.hpp>

class RegionDetector : public QObject
{
    Q_OBJECT

public:
    explicit RegionDetector(QObject *parent = nullptr);
    
    QVector<QRect> detectRegions(const cv::Mat &inputImage);
    QVector<QRect> detectRegionsWithBias(const cv::Mat &inputImage, int bias = 10);
    
    void setThreshold(int threshold);
    void setMinArea(int minArea);
    void setMaxArea(int maxArea);

private:
    int m_threshold;
    int m_minArea;
    int m_maxArea;
    
    QRect expandRect(const cv::Rect &rect, int bias, const cv::Size &imageSize);
};

#endif // REGIONDETECTOR_H
