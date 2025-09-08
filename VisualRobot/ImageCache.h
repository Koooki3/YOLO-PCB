#pragma once
#include <opencv2/core/mat.hpp>
#include <string>
#include <unordered_map>
#include <mutex>

class ImageCache {
public:
    // 针对8GB内存系统的配置
    static constexpr size_t MAX_CACHE_SIZE = 1024 * 1024 * 1024; // 1GB 最大缓存大小
    static constexpr size_t MAX_CACHE_ITEMS = 20; // 最大缓存项数量
    
    static ImageCache& getInstance() {
        static ImageCache instance;
        return instance;
    }

    bool get(const std::string& key, cv::Mat& result) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = cache_.find(key);
        if(it != cache_.end()) {
            result = it->second.clone();
            return true;
        }
        return false;
    }

    void put(const std::string& key, const cv::Mat& image) {
        std::lock_guard<std::mutex> lock(mutex_);
        
        // 计算当前缓存大小
        size_t currentSize = 0;
        for(const auto& pair : cache_) {
            currentSize += pair.second.total() * pair.second.elemSize();
        }
        
        // 检查是否需要清理缓存
        while(!cache_.empty() && 
              (cache_.size() >= MAX_CACHE_ITEMS || 
               currentSize + (image.total() * image.elemSize()) > MAX_CACHE_SIZE)) {
            // 移除最早的缓存项
            auto it = cache_.begin();
            currentSize -= it->second.total() * it->second.elemSize();
            cache_.erase(it);
        }
        
        // 添加新的缓存项
        cache_[key] = image.clone();
    }

    void clear() {
        std::lock_guard<std::mutex> lock(mutex_);
        cache_.clear();
    }

private:
    ImageCache() = default;
    ~ImageCache() = default;
    ImageCache(const ImageCache&) = delete;
    ImageCache& operator=(const ImageCache&) = delete;

    std::unordered_map<std::string, cv::Mat> cache_;
    std::mutex mutex_;
};
