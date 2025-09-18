#include "Format.h"
#include <QImage>
#include <halconcpp/HalconCpp.h>
#include <Halcon.h>
#include <halconcpp/HDevThread.h>
#include <vector>
#include <stdexcept>
#include <QDebug>

using namespace HalconCpp;
using namespace std;

/*提示：从HImage对象开始完成转换到QImage对象再转换回HImage对象经测试是完备的
 *     反之则不完备会出现图像错误（该Bug预计后期修复）
 *建议：调用Halcon读取图像成HImage再转成QImage，这样使用两个图像对象不会出现
 *     预期之外的错误
 */

HImage QImageToHImage(const QImage &qImage)
{
    if (qImage.isNull() || qImage.width() <= 0 || qImage.height() <= 0)
    {
        throw invalid_argument("Invalid QImage");
    }

    // 转换为32位RGB格式
    QImage converted = qImage.convertToFormat(QImage::Format_RGB32);
    int width = converted.width();
    int height = converted.height();

    // 准备Halcon三通道数据
    vector<unsigned char> r(width * height);
    vector<unsigned char> g(width * height);
    vector<unsigned char> b(width * height);

    // 提取RGB分量: 直接使用constBits()访问BGRA数据
    const unsigned char* bits = converted.constBits();
    for (int y = 0; y < height; ++y)
    {
        const unsigned char* line = bits + y * converted.bytesPerLine();
        for (int x = 0; x < width; ++x)
        {
            int index = y * width + x;
            b[index] = line[0];
            g[index] = line[1];
            r[index] = line[2];
            // 忽略A
            line += 4;
        }
    }

    // 创建Halcon图像
    HImage himg;
    if (qImage.format() == QImage::Format_Grayscale8)
    {
        // 对于灰度图直接用单通道
        himg.GenImage1("byte", width, height, r.data());
    }
    else
    {
        himg.GenImage3("byte", width, height, r.data(), g.data(), b.data());
    }

    return himg;
}

QImage HImageToQImage(const HImage &hImage)
{
    // 检查有效性
    HTuple channels = hImage.CountChannels();
    if (channels[0].I() != 3 && channels[0].I() != 1)
    {
        throw invalid_argument("HImage must have 1 or 3 channels");
    }

    // 获取图像参数
    HTuple widthTuple, heightTuple;
    hImage.GetImageSize(&widthTuple, &heightTuple);
    if (widthTuple[0].I() <= 0 || heightTuple[0].I() <= 0)
    {
        throw invalid_argument("Invalid HImage size");
    }
    Hlong w = widthTuple[0].I();
    Hlong h = heightTuple[0].I();

    // 如果已经是单通道（灰度），直接处理
    QImage::Format qtFormat = (channels[0].I() == 1) ? QImage::Format_Grayscale8 : QImage::Format_RGB888;
    int bpp = (channels[0].I() == 1) ? 1 : 3;

    // 转换为交错RGB格式
    HImage processed = hImage;
    if (channels[0].I() == 3)
    {
        processed = hImage.InterleaveChannels("rgb", "match", 255);
    }

    // 获取指针
    HString type;
    void* ptr = processed.GetImagePointer1(&type, &w, &h);
    if (type != "byte")
    {
        throw runtime_error("Unsupported image type: " + string(type.Text()));
    }
    const unsigned char* data = reinterpret_cast<const unsigned char*>(ptr);

    // 计算每行字节数并处理对齐问题
    const int unpaddedBytesPerLine = w * bpp;

    // 创建目标图像缓冲区（初始化为0自动处理填充）
    QImage result(w, h, qtFormat);
    result.fill(0);

    // 逐行复制数据（跳过填充区域）
    for (Hlong y = 0; y < h; ++y)
    {
        const unsigned char* src = data + y * unpaddedBytesPerLine;
        unsigned char* dst = result.scanLine(y);
        memcpy(dst, src, unpaddedBytesPerLine);
    }

    return result;
}

bool saveHImageWithHalcon(const HImage& image, const QString& filePath, const QString& format, int quality)
{
    try
    {
        // 转换路径和格式
        QByteArray pathBytes = filePath.toUtf8();
        HTuple hFilePath(pathBytes.constData());
        HTuple hFormat(format.toStdString().c_str());

        // 处理质量参数
        if (format.compare("jpeg", Qt::CaseInsensitive) == 0 && quality >= 0)
        {
            // 对于 JPEG 且有指定质量
            HTuple hQuality(quality);
            image.WriteImage(hFormat, hQuality, hFilePath);
        }
        else
        {
            // 其他格式或无质量要求
            image.WriteImage(hFormat, HTuple(), hFilePath);
        }
        return true;
    }
    catch (HException& e)
    {
        qDebug() << "Halcon Error:" << e.ErrorMessage().Text();
        return false;
    }
}
