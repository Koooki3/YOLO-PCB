// === 模板检测：新增成员 ===
private:
    cv::Mat m_templateGray;          // 模板灰度图
    bool    m_hasTemplate = false;   // 是否已有模板
    double  m_diffThresh = 25.0;     // 差异二值阈值（配准后的 absdiff 后再高斯平滑）
    double  m_minDefectArea = 1200;  // 过滤小区域（像素），按你的分辨率可调
    int     m_orbFeatures = 1500;    // ORB特征点数量（配准用）

    // 将缓存的最新一帧转为BGR Mat（经SDK内存编码为JPEG后imdecode，稳妥）
    bool grabLastFrameBGR(cv::Mat& outBGR);

    // Mat 转 QPixmap（显示用）
    static QPixmap matToQPixmap(const cv::Mat& bgr);

    // 把当前帧设为模板（从当前帧或文件）
    bool setTemplateFromCurrent();
    bool setTemplateFromFile(const QString& path);

    // 配准：计算 H（模板 <- 当前）
    bool computeHomography(const cv::Mat& curGray, cv::Mat& H, std::vector<cv::DMatch>* dbgMatches=nullptr);

    // 检测：根据配准后差异，得到在“当前图像坐标系”的缺陷外接框
    std::vector<cv::Rect> detectDefects(const cv::Mat& curBGR, const cv::Mat& H, cv::Mat* dbgMask=nullptr);

private slots:
    void on_setTemplate_clicked();   // 可选：从当前帧设模板（或弹框选择文件）
    void on_detect_clicked();        // 你要的检测按钮


// ====== Mat 转 QPixmap ======
QPixmap MainWindow::matToQPixmap(const cv::Mat& bgr)
{
    if (bgr.empty()) return QPixmap();
    cv::Mat rgb;
    cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
    QImage img(rgb.data, rgb.cols, rgb.rows, rgb.step, QImage::Format_RGB888);
    return QPixmap::fromImage(img.copy());
}

// ====== 从缓存的最后一帧取图：BGR ======
bool MainWindow::grabLastFrameBGR(cv::Mat& outBGR)
{
    vector<unsigned char> frameCopy;
    MV_FRAME_OUT_INFO_EX info{};
    {
        std::lock_guard<std::mutex> lk(m_frameMtx);
        if (!m_hasFrame || m_lastFrame.empty()) {
            appendLog("暂无可用图像，请先开始采集。", WARNNING);
            return false;
        }
        frameCopy = m_lastFrame;
        info      = m_lastInfo;
    }

    // 用 SDK 把原始帧编码成 JPEG，然后用 OpenCV 解码得到BGR
    unsigned int dstMax = info.nWidth * info.nHeight * 3 + 4096;
    std::unique_ptr<unsigned char[]> pDst(new (std::nothrow) unsigned char[dstMax]);
    if (!pDst) {
        appendLog("抓帧转换：内存不足（编码缓冲）", ERROR);
        return false;
    }

    MV_SAVE_IMAGE_PARAM_EX3 save{};
    save.enImageType   = MV_Image_Jpeg;
    save.enPixelType   = info.enPixelType;
    save.nWidth        = info.nWidth;
    save.nHeight       = info.nHeight;
    save.nDataLen      = info.nFrameLen;
    save.pData         = frameCopy.data();
    save.pImageBuffer  = pDst.get();
    save.nImageLen     = dstMax;
    save.nJpgQuality   = 90;

    int nRet = m_pcMyCamera ? m_pcMyCamera->SaveImage(&save) : MV_E_HANDLE;
    if (MV_OK != nRet || save.nImageLen == 0) {
        appendLog("抓帧转换失败（编码阶段）", ERROR);
        return false;
    }

    // OpenCV 解码
    cv::Mat encoded(1, static_cast<int>(save.nImageLen), CV_8U, pDst.get());
    cv::Mat bgr = cv::imdecode(encoded, cv::IMREAD_COLOR);
    if (bgr.empty()) {
        appendLog("imdecode 失败", ERROR);
        return false;
    }
    outBGR = bgr.clone();
    return true;
}

// ====== 将当前帧设为模板 ======
bool MainWindow::setTemplateFromCurrent()
{
    cv::Mat bgr;
    if (!grabLastFrameBGR(bgr)) return false;

    cv::Mat gray;
    cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);
    cv::GaussianBlur(gray, gray, cv::Size(3,3), 0);
    m_templateGray = gray.clone();
    m_hasTemplate  = true;

    appendLog(QString("已将当前帧设为模板，尺寸：%1x%2")
              .arg(m_templateGray.cols).arg(m_templateGray.rows), INFO);
    return true;
}

bool MainWindow::setTemplateFromFile(const QString& path)
{
    cv::Mat bgr = cv::imread(path.toStdString(), cv::IMREAD_COLOR);
    if (bgr.empty()) {
        appendLog("读取模板文件失败：" + path, ERROR);
        return false;
    }
    cv::Mat gray;
    cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);
    cv::GaussianBlur(gray, gray, cv::Size(3,3), 0);
    m_templateGray = gray.clone();
    m_hasTemplate  = true;

    appendLog(QString("已从文件加载模板：%1（%2x%3）")
              .arg(path).arg(m_templateGray.cols).arg(m_templateGray.rows), INFO);
    return true;
}

// ====== 计算单应性（模板 <- 当前） ======
bool MainWindow::computeHomography(const cv::Mat& curGray, cv::Mat& H, std::vector<cv::DMatch>* dbgMatches)
{
    if (!m_hasTemplate || m_templateGray.empty()) return false;

    // ORB 特征
    cv::Ptr<cv::ORB> orb = cv::ORB::create(m_orbFeatures);
    std::vector<cv::KeyPoint> kptT, kptC;
    cv::Mat desT, desC;
    orb->detectAndCompute(m_templateGray, cv::noArray(), kptT, desT);
    orb->detectAndCompute(curGray,       cv::noArray(), kptC, desC);

    if (desT.empty() || desC.empty()) {
        appendLog("配准失败：未检测到足够特征。", ERROR);
        return false;
    }

    // 暴力匹配 + 交叉检验
    cv::BFMatcher matcher(cv::NORM_HAMMING, true);
    std::vector<cv::DMatch> matches;
    matcher.match(desT, desC, matches);
    if (matches.size() < 8) {
        appendLog("配准失败：匹配对过少。", ERROR);
        return false;
    }

    // 根据距离剔除离群
    double maxDist = 0, minDist = 1e9;
    for (auto& m : matches) {
        double d = m.distance;
        maxDist = std::max(maxDist, d);
        minDist = std::min(minDist, d);
    }
    std::vector<cv::DMatch> good;
    double thr = std::max(2.0*minDist, 30.0); // 经验阈值
    for (auto& m : matches) if (m.distance <= thr) good.push_back(m);
    if (good.size() < 8) good = matches; // 兜底

    if (dbgMatches) *dbgMatches = good;

    std::vector<cv::Point2f> ptsT, ptsC;
    ptsT.reserve(good.size());
    ptsC.reserve(good.size());
    for (auto& m : good) {
        ptsT.push_back(kptT[m.queryIdx].pt);
        ptsC.push_back(kptC[m.trainIdx].pt);
    }

    // RANSAC 求 H（模板 <- 当前）
    std::vector<unsigned char> inliers;
    H = cv::findHomography(ptsC, ptsT, cv::RANSAC, 3.0, inliers);
    if (H.empty()) {
        appendLog("配准失败：单应矩阵为空。", ERROR);
        return false;
    }
    appendLog(QString("配准成功：内点数 %1 / %2").arg(std::count(inliers.begin(), inliers.end(), 1)).arg((int)ptsT.size()), INFO);
    return true;
}

// ====== 检测核心：返回在“当前图像坐标系”的外接框 ======
std::vector<cv::Rect> MainWindow::detectDefects(const cv::Mat& curBGR, const cv::Mat& H, cv::Mat* dbgMask)
{
    std::vector<cv::Rect> boxes;

    // 统一到模板坐标系做差异
    cv::Mat curGray;
    cv::cvtColor(curBGR, curGray, cv::COLOR_BGR2GRAY);
    cv::GaussianBlur(curGray, curGray, cv::Size(3,3), 0);

    cv::Mat warped;
    cv::warpPerspective(curGray, warped, H, m_templateGray.size(), cv::INTER_LINEAR);

    // 差异图
    cv::Mat diff;
    cv::absdiff(m_templateGray, warped, diff);
    cv::GaussianBlur(diff, diff, cv::Size(5,5), 0);

    // 二值化（可改Otsu）
    cv::Mat bin;
    if (m_diffThresh <= 0) {
        cv::threshold(diff, bin, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
    } else {
        cv::threshold(diff, bin, m_diffThresh, 255, cv::THRESH_BINARY);
    }

    // 形态学净化
    cv::morphologyEx(bin, bin, cv::MORPH_OPEN, cv::getStructuringElement(cv::MORPH_RECT, {5,5}));
    cv::morphologyEx(bin, bin, cv::MORPH_CLOSE, cv::getStructuringElement(cv::MORPH_RECT, {9,9}));

    if (dbgMask) *dbgMask = bin.clone();

    // 找轮廓（在模板坐标系）
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(bin, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    if (contours.empty()) return boxes;

    // 将模板坐标系下的外接框四角回投影到“当前图像坐标系”
    cv::Mat Hinv;
    if (!cv::invert(H, Hinv)) {
        appendLog("单应矩阵不可逆，无法回投影。", ERROR);
        return boxes;
    }

    for (auto& c : contours) {
        double area = cv::contourArea(c);
        if (area < m_minDefectArea) continue;

        cv::Rect r = cv::boundingRect(c); // 模板系下
        // 四角
        std::vector<cv::Point2f> srcPts = {
            { (float)r.x, (float)r.y },
            { (float)(r.x + r.width), (float)r.y },
            { (float)(r.x + r.width), (float)(r.y + r.height) },
            { (float)r.x, (float)(r.y + r.height) }
        };
        std::vector<cv::Point2f> dstPts;
        cv::perspectiveTransform(srcPts, dstPts, Hinv);

        // 用回投影后的四角做外接框（当前图像坐标系）
        float minx = curBGR.cols, miny = curBGR.rows, maxx = 0, maxy = 0;
        for (auto& p : dstPts) {
            minx = std::min(minx, p.x); miny = std::min(miny, p.y);
            maxx = std::max(maxx, p.x); maxy = std::max(maxy, p.y);
        }
        cv::Rect box(cv::Point2f(minx, miny), cv::Point2f(maxx, maxy));
        box &= cv::Rect(0,0, curBGR.cols, curBGR.rows); // 裁边
        if (box.area() > 0) boxes.push_back(box);
    }

    return boxes;
}

// ====== 可选：设置模板按钮 ======
void MainWindow::on_setTemplate_clicked()
{
    // 弹窗选择：当前帧 / 从文件
    QMessageBox msg(this);
    msg.setWindowTitle("设置模板");
    msg.setText("选择模板来源：");
    auto *btnCur = msg.addButton("使用当前帧", QMessageBox::AcceptRole);
    auto *btnFile= msg.addButton("从文件选择...", QMessageBox::ActionRole);
    msg.addButton(QMessageBox::Cancel);

    msg.exec();
    QAbstractButton* clicked = msg.clickedButton();
    if (clicked == btnCur) {
        if (setTemplateFromCurrent())
            appendLog("模板已更新（来自当前帧）", INFO);
    } else if (clicked == btnFile) {
        QString path = QFileDialog::getOpenFileName(this, "选择模板图像", ".", "Images (*.png *.jpg *.jpeg *.bmp)");
        if (!path.isEmpty() && setTemplateFromFile(path))
            appendLog("模板已更新（来自文件）", INFO);
    }
}

// ====== 你要的：缺陷检测按钮 ======
void MainWindow::on_detect_clicked()
{
    if (!m_hasTemplate) {
        appendLog("尚未设置模板，请先设置模板。", WARNNING);
        // 尝试直接用当前帧设模板，继续流程（也可直接 return）
        if (!setTemplateFromCurrent()) return;
    }

    // 取当前帧
    cv::Mat curBGR;
    if (!grabLastFrameBGR(curBGR)) return;

    // 计算配准 H（模板 <- 当前）
    cv::Mat curGray;
    cv::cvtColor(curBGR, curGray, cv::COLOR_BGR2GRAY);
    cv::GaussianBlur(curGray, curGray, cv::Size(3,3), 0);

    cv::Mat H;
    if (!computeHomography(curGray, H)) return;

    // 检测
    QElapsedTimer t; t.start();
    cv::Mat dbgMask;
    auto boxes = detectDefects(curBGR, H, &dbgMask);
    appendLog(QString("模板检测耗时: %1 ms，候选缺陷框数: %2")
              .arg(t.elapsed()).arg((int)boxes.size()), INFO);

    // 绘制结果
    cv::Mat draw = curBGR.clone();
    for (auto& b : boxes) {
        cv::rectangle(draw, b, cv::Scalar(0,0,255), 2); // 红框
    }

    // 展示到 widgetDisplay_2
    QPixmap pm = matToQPixmap(draw);
    if (!pm.isNull()) {
        // 自适应显示
        QPixmap scaled = pm.scaled(ui->widgetDisplay_2->size(),
                                   Qt::KeepAspectRatio, Qt::SmoothTransformation);
        ui->widgetDisplay_2->setPixmap(scaled);
        ui->widgetDisplay_2->setAlignment(Qt::AlignCenter);
        appendLog("缺陷结果已显示。", INFO);
    } else {
        appendLog("显示失败：QPixmap 为空。", ERROR);
    }

    // 日志每个框
    for (size_t i=0; i<boxes.size(); ++i) {
        appendLog(QString("缺陷框 %1: (x=%2, y=%3, w=%4, h=%5)")
                  .arg(i+1).arg(boxes[i].x).arg(boxes[i].y)
                  .arg(boxes[i].width).arg(boxes[i].height), INFO);
    }

    // 如需同时叠加尺寸/角度的浮窗，可复用你已有的 drawOverlayOnDisplay2()
}
