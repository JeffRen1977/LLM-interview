// 智能眼镜端侧 AI Runtime —— Hybrid 路由决策 + KV Cache LRU 分块回收 参考实现
//
// 编译: g++ -std=c++17 -O2 smart_glasses_runtime.cpp -o smart_glasses_runtime
// 运行: ./smart_glasses_runtime
//
// 两个可运行的系统件:
//   1. HybridRouter        : 启发式端-云路由决策 (弱网强制本地 / 复杂任务上云)。
//   2. KVCacheBlockManager : 静态预分配分块 + LRU 回收, 端侧绝不动态 new/malloc。

#include <cstdint>
#include <iostream>
#include <list>
#include <string>
#include <unordered_map>
#include <vector>

// ─────────────────────────────────────────────────────────────
// 1. 端-云协同动态路由决策器
// ─────────────────────────────────────────────────────────────
enum class InferenceRoute {
    LOCAL_NPU,   // 纯本地: 查时间、拍照、简单实体识别
    CLOUD_LMM    // 云端: 长翻译、复杂逻辑推理
};

const char* to_string(InferenceRoute r) {
    return r == InferenceRoute::LOCAL_NPU ? "LOCAL_NPU" : "CLOUD_LMM";
}

InferenceRoute make_routing_decision(const std::string& intent, double current_rtt_ms) {
    // 1) 网络检测: 延迟过高或断网 -> 强制本地兜底
    if (current_rtt_ms > 300.0) {
        return InferenceRoute::LOCAL_NPU;
    }
    // 2) 意图分析: 复杂/长上下文任务 -> 上云
    if (intent == "complex_reasoning" || intent == "long_translation") {
        return InferenceRoute::CLOUD_LMM;
    }
    // 3) 默认本地, 省电低延迟
    return InferenceRoute::LOCAL_NPU;
}

// ─────────────────────────────────────────────────────────────
// 2. KV Cache 分块管理器: 静态预分配 + LRU 回收
//    - 初始化时一次性占满物理显存, 运行期零动态分配。
//    - 每个 Block 固定大小; 空闲块用 free list; 占用块进 LRU 链表。
//    - 显存满时自动驱逐最久未用的块 (LRU), 杜绝 OOM。
// ─────────────────────────────────────────────────────────────
class KVCacheBlockManager {
public:
    KVCacheBlockManager(size_t block_count, size_t block_bytes)
        : block_bytes_(block_bytes) {
        // 一次性预分配整块物理内存 (模拟 NPU 显存 Arena)
        arena_.resize(block_count * block_bytes);
        for (size_t i = 0; i < block_count; ++i) {
            free_list_.push_back(i);
        }
    }

    // 为某个 token 序列位置申请一个 block。显存满则 LRU 驱逐。
    // 返回 block 内的可写指针 (零拷贝: 直接指向预分配 Arena)。
    uint8_t* acquire(uint64_t seq_key) {
        size_t idx;
        if (!free_list_.empty()) {
            idx = free_list_.back();
            free_list_.pop_back();
        } else {
            // 无空闲块 -> 驱逐 LRU (链表尾部 = 最久未用)
            idx = evict_lru();
            ++evictions_;
        }
        // 挂到 LRU 头部 (最近使用)
        lru_.push_front(idx);
        block_of_key_[seq_key] = lru_.begin();
        key_of_block_[idx] = seq_key;
        return &arena_[idx * block_bytes_];
    }

    // 命中已有 key: 提升为最近使用 (move-to-front)
    bool touch(uint64_t seq_key) {
        auto it = block_of_key_.find(seq_key);
        if (it == block_of_key_.end()) return false;
        lru_.splice(lru_.begin(), lru_, it->second);  // O(1) 移到头部
        it->second = lru_.begin();
        return true;
    }

    size_t free_blocks() const { return free_list_.size(); }
    size_t used_blocks() const { return lru_.size(); }
    uint64_t evictions() const { return evictions_; }

private:
    size_t evict_lru() {
        size_t victim = lru_.back();
        lru_.pop_back();
        uint64_t victim_key = key_of_block_[victim];
        block_of_key_.erase(victim_key);
        key_of_block_.erase(victim);
        return victim;
    }

    size_t block_bytes_;
    std::vector<uint8_t> arena_;                 // 预分配显存
    std::vector<size_t> free_list_;              // 空闲块索引
    std::list<size_t> lru_;                      // 占用块 LRU (front=最新)
    std::unordered_map<uint64_t, std::list<size_t>::iterator> block_of_key_;
    std::unordered_map<size_t, uint64_t> key_of_block_;
    uint64_t evictions_ = 0;
};

int main() {
    // ---- 路由决策演示 ----
    struct Case { std::string intent; double rtt; };
    std::vector<Case> cases = {
        {"get_time", 20.0},            // 简单 + 好网 -> 本地
        {"long_translation", 50.0},    // 复杂 + 好网 -> 云端
        {"complex_reasoning", 500.0},  // 复杂但弱网 -> 强制本地
        {"take_photo", 10.0},          // 简单 -> 本地
    };
    std::cout << "== Hybrid Router ==\n";
    for (const auto& c : cases) {
        std::cout << "  intent=" << c.intent << ", rtt=" << c.rtt
                  << "ms -> " << to_string(make_routing_decision(c.intent, c.rtt))
                  << "\n";
    }

    // ---- KV Cache LRU 分块回收演示 ----
    std::cout << "\n== KV Cache Block Manager (4 blocks, static) ==\n";
    KVCacheBlockManager kv(/*block_count=*/4, /*block_bytes=*/1024);

    // 模拟解码: 生成 8 个 token, 每个占一个 block; 显存只有 4 块 -> LRU 驱逐
    for (uint64_t token = 0; token < 8; ++token) {
        uint8_t* blk = kv.acquire(token);
        blk[0] = static_cast<uint8_t>(token);  // 零拷贝写入预分配显存
        std::cout << "  gen token " << token
                  << " | used=" << kv.used_blocks()
                  << " free=" << kv.free_blocks()
                  << " evictions=" << kv.evictions() << "\n";
    }

    std::cout << "\nTotal evictions: " << kv.evictions()
              << " (预期 4: 前 4 个 token 被 LRU 驱逐)\n";
    return 0;
}
