#ifndef HALCON_H
#define HALCON_H

#include <QImage>
#include <QString>
#include <QVector>
#include <QPointF>
#include <string>
#include <vector>

// Halcon 头文件
#include <halconcpp/HalconCpp.h>
#include <Halcon.h>
#include <halconcpp/HDevThread.h>

using namespace HalconCpp;
using namespace std;

// 图像转换函数
HImage QImageToHImage(const QImage &qImage);
QImage HImageToQImage(const HImage &hImage);
bool saveHImageWithHalcon(const HImage& image, const QString& filePath, const QString& format, int quality);

// 图像处理算法函数
int Algorithm(const string& imgPath, HTuple& Row, HTuple& Col);
int getCoords(QVector<QPointF>& WorldCoord, QVector<QPointF>& PixelCoord, double size);

#endif // HALCON_H
