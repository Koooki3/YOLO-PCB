#pragma once
#include <opencv2/core/mat.hpp>
#include <vector>
#include <mutex>

class ImageMemoryPool {
public:
    // 针对8GB内存系统的配置
    static constexpr size_t MAX_POOL_SIZE = 512 * 1024 * 1024; // 512MB 最大内存池大小
    static constexpr size_t MAX_BUFFER_COUNT = 32; // 最大缓冲区数量

    static ImageMemoryPool& getInstance() {
        static ImageMemoryPool instance;
        return instance;
    }

    cv::Mat acquire(int rows, int cols, int type) {
        std::lock_guard<std::mutex> lock(mutex_);
        
        // 查找合适大小的缓存
        for(auto it = pool_.begin(); it != pool_.end(); ++it) {
            if(it->rows == rows && it->cols == cols && it->type() == type) {
                cv::Mat mat = *it;
                pool_.erase(it);
                return mat;
            }
        }
        
        // 没有找到合适的，创建新的
        return cv::Mat(rows, cols, type);
    }

    void release(cv::Mat& mat) {
        if(!mat.empty()) {
            std::lock_guard<std::mutex> lock(mutex_);
            
            // 计算当前内存池大小
            size_t currentSize = 0;
            for(const auto& m : pool_) {
                currentSize += m.total() * m.elemSize();
            }
            
            // 检查是否超出限制
            if(pool_.size() < MAX_BUFFER_COUNT && 
               currentSize + (mat.total() * mat.elemSize()) <= MAX_POOL_SIZE) {
                pool_.push_back(mat);
            } else {
                // 如果超出限制，直接释放
                mat.release();
            }
        }
    }

private:
    ImageMemoryPool() = default;
    ~ImageMemoryPool() = default;
    ImageMemoryPool(const ImageMemoryPool&) = delete;
    ImageMemoryPool& operator=(const ImageMemoryPool&) = delete;

    std::vector<cv::Mat> pool_;
    std::mutex mutex_;
};
