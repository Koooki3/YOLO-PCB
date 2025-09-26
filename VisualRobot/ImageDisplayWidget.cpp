#include "ImageDisplayWidget.h"
#include <QPainter>
#include <QPen>
#include <QDebug>

ImageDisplayWidget::ImageDisplayWidget(QWidget *parent) : QWidget(parent)
{
    m_borderThickness = 2;
    m_borderColor = QColor(0, 255, 0); // 绿色，参考DIP.cpp中的颜色
    updateBorderStyle();
}

void ImageDisplayWidget::setImage(const QImage &image)
{
    m_image = image;
    update();
}

void ImageDisplayWidget::setImage(const cv::Mat &mat)
{
    if (mat.empty()) {
        m_image = QImage();
        update();
        return;
    }

    // 转换OpenCV Mat到QImage
    cv::Mat temp;
    if (mat.channels() == 3) {
        cv::cvtColor(mat, temp, cv::COLOR_BGR2RGB);
        m_image = QImage(temp.data, temp.cols, temp.rows, temp.step, QImage::Format_RGB888);
    } else if (mat.channels() == 1) {
        cv::cvtColor(mat, temp, cv::COLOR_GRAY2RGB);
        m_image = QImage(temp.data, temp.cols, temp.rows, temp.step, QImage::Format_RGB888);
    } else {
        m_image = QImage();
    }
    
    update();
}

void ImageDisplayWidget::clearRegions()
{
    m_regions.clear();
    update();
}

void ImageDisplayWidget::addRegion(const QRect &region)
{
    m_regions.append(region);
    update();
}

void ImageDisplayWidget::setRegions(const QVector<QRect> &regions)
{
    m_regions = regions;
    update();
}

void ImageDisplayWidget::paintEvent(QPaintEvent *event)
{
    Q_UNUSED(event);
    
    QPainter painter(this);
    painter.setRenderHint(QPainter::Antialiasing);
    
    // 绘制背景
    painter.fillRect(rect(), Qt::black);
    
    // 如果没有图像，直接返回
    if (m_image.isNull()) {
        return;
    }
    
    // 计算缩放比例，保持宽高比
    QRect targetRect = rect();
    QImage scaledImage = m_image.scaled(targetRect.size(), Qt::KeepAspectRatio, Qt::SmoothTransformation);
    
    // 计算居中位置
    int x = (targetRect.width() - scaledImage.width()) / 2;
    int y = (targetRect.height() - scaledImage.height()) / 2;
    QRect imageRect(x, y, scaledImage.width(), scaledImage.height());
    
    // 绘制图像
    painter.drawImage(imageRect, scaledImage);
    
    // 绘制检测区域框
    if (!m_regions.isEmpty()) {
        QPen pen(m_borderColor);
        pen.setWidth(m_borderThickness);
        painter.setPen(pen);
        
        // 计算缩放比例
        double scaleX = static_cast<double>(scaledImage.width()) / m_image.width();
        double scaleY = static_cast<double>(scaledImage.height()) / m_image.height();
        
        for (const QRect &region : m_regions) {
            // 缩放矩形框到显示尺寸
            QRect scaledRegion(
                x + static_cast<int>(region.x() * scaleX),
                y + static_cast<int>(region.y() * scaleY),
                static_cast<int>(region.width() * scaleX),
                static_cast<int>(region.height() * scaleY)
            );
            
            // 绘制矩形框
            painter.drawRect(scaledRegion);
        }
    }
}

void ImageDisplayWidget::updateBorderStyle()
{
    // 根据DIP.cpp中的实现，线宽基于图像宽度计算
    // 这里我们使用固定值，因为实时显示时图像尺寸可能变化
    // 实际使用时可以根据需要动态调整
    m_borderThickness = 2; // 参考DIP.cpp中的实现
    m_borderColor = QColor(0, 255, 0); // 绿色
}
