#include "Halcon.h"
#include <QImage>
#include <QDebug>
#include <QFileDialog>
#include <QInputDialog>
#include <QCoreApplication>
#include <vector>
#include <stdexcept>

using namespace HalconCpp;
using namespace std;

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

int Algorithm(const string& imgPath, HTuple& Row, HTuple& Col)
{
    try
    {
        HImage image;
        image.ReadImage(HTuple(imgPath.c_str()));

        HRegion region;
        HImage redChannel = image.Decompose3().G();
        region = redChannel.Threshold(0, 100);

        HRegion connectedRegions = region.Connection();
        HRegion selectedRegions = connectedRegions.SelectShape("area", "and", 5000, 9999999);

        if (selectedRegions.CountObj() == 0)
        {
            return 1; // 没有找到区域
        }

        HXLDCont contours = selectedRegions.GenContourRegionXld("border");
        HXLDCont corners = contours.CornerHarris(0.01, 1, 0.01, 3, 0.04);

        HTuple cornerRow, cornerCol;
        corners.GetContourXld(&cornerRow, &cornerCol);

        if (cornerRow.Length() == 0 || cornerCol.Length() == 0)
        {
            return 2; // 没有找到角点
        }

        Row = cornerRow;
        Col = cornerCol;

        return 0; // 成功
    }
    catch (HException& e)
    {
        qDebug() << "Halcon Error in Algorithm:" << e.ErrorMessage().Text();
        return -1; // Halcon异常
    }
}

int getCoords(QVector<QPointF>& WorldCoord, QVector<QPointF>& PixelCoord, double size)
{
    try
    {
        // 清空输入向量
        WorldCoord.clear();
        PixelCoord.clear();

        // 打开图像文件对话框
        QString fileName = QFileDialog::getOpenFileName(nullptr,
            "选择标定板图像",
            QCoreApplication::applicationDirPath(),
            "图像文件 (*.png *.jpg *.bmp *.tiff)");

        if (fileName.isEmpty())
        {
            return 1; // 用户取消
        }

        // 读取图像
        HImage image;
        image.ReadImage(HTuple(fileName.toStdString().c_str()));

        // 检测角点
        HTuple rows, cols;
        int result = Algorithm(fileName.toStdString(), rows, cols);
        if (result != 0)
        {
            return result + 10; // Algorithm错误代码偏移
        }

        if (rows.Length() < 4 || cols.Length() < 4)
        {
            return 3; // 角点数量不足
        }

        // 获取图像大小
        HTuple width, height;
        image.GetImageSize(&width, &height);

        // 显示图像和角点
        HWindow window(0, 0, width[0].I(), height[0].I());
        window.SetPart(0, 0, height[0].I() - 1, width[0].I() - 1);
        window.DispImage(image);
        window.SetColor("red");
        for (int i = 0; i < rows.Length(); i++)
        {
            window.DispCross(rows[i].D(), cols[i].D(), 36, 0);
        }
        window.DumpWindow("jpeg", "detected_corners.jpg");

        // 提示用户输入世界坐标
        for (int i = 0; i < rows.Length(); i++)
        {
            bool ok;
            double x = QInputDialog::getDouble(nullptr,
                "输入世界坐标",
                QString("请输入第 %1 个角点的 X 坐标 (mm):").arg(i + 1),
                0.0, -10000.0, 10000.0, 2, &ok);
            if (!ok)
            {
                return 4; // 用户取消输入
            }

            double y = QInputDialog::getDouble(nullptr,
                "输入世界坐标",
                QString("请输入第 %1 个角点的 Y 坐标 (mm):").arg(i + 1),
                0.0, -10000.0, 10000.0, 2, &ok);
            if (!ok)
            {
                return 4; // 用户取消输入
            }

            WorldCoord.append(QPointF(x, y));
            PixelCoord.append(QPointF(cols[i].D(), rows[i].D()));
        }

        return 0; // 成功
    }
    catch (HException& e)
    {
        qDebug() << "Halcon Error in getCoords:" << e.ErrorMessage().Text();
        return -1; // Halcon异常
    }
}
