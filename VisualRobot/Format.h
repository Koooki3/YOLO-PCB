/************************************************************************/
/* 提供Halcon的HImage类对象与Qt的QImage类对象相互转换的支持 */
/************************************************************************/

#ifndef FORMAT_H
#define FORMAT_H

#include <QImage>
#include <halconcpp/HalconCpp.h>
#include <Halcon.h>
#include <halconcpp/HDevThread.h>

using namespace HalconCpp;
using namespace std;


HImage QImageToHImage(const QImage &qImage);

QImage HImageToQImage(const HImage &hImage);

bool saveHImageWithHalcon(const HImage& image, const QString& filePath, const QString& format, int quality);

#endif // FORMAT_H
