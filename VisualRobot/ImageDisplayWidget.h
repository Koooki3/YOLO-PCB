#ifndef IMAGEDISPLAYWIDGET_H
#define IMAGEDISPLAYWIDGET_H

#include <QWidget>
#include <QImage>
#include <QPixmap>
#include <QPainter>
#include <QVector>
#include <QRect>
#include <opencv2/opencv.hpp>

class ImageDisplayWidget : public QWidget
{
    Q_OBJECT

public:
    explicit ImageDisplayWidget(QWidget *parent = nullptr);
    
    void setImage(const QImage &image);
    void setImage(const cv::Mat &mat);
    void clearRegions();
    void addRegion(const QRect &region);
    void setRegions(const QVector<QRect> &regions);

protected:
    void paintEvent(QPaintEvent *event) override;

private:
    QImage m_image;
    QVector<QRect> m_regions;
    int m_borderThickness;
    QColor m_borderColor;
    
    void updateBorderStyle();
};

#endif // IMAGEDISPLAYWIDGET_H
