// 端侧 Token 缓存内存池 —— 简化教学版
// 编译: g++ -std=c++17 -O2 -pthread token_memory_pool.cpp -o token_memory_pool
//
// 核心只有四步:
//   1. 预分配 Arena, 切成定长 Block
//   2. 空闲 Block 用无锁 free list 串联 (CAS)
//   3. allocate 用 placement new 就地构造
//   4. unique_ptr + 自定义 Deleter 自动回收 (RAII)

#include <atomic>
#include <iostream>
#include <memory>

template <typename T, size_t N>
class TokenMemoryPool {
    union Block {
        T data;
        Block* next;
        Block() {}
        ~Block() {}
    };

public:
    struct Deleter {
        void operator()(T* p) const {
            if (p) TokenMemoryPool::instance().recycle(p);
        }
    };
    using Ptr = std::unique_ptr<T, Deleter>;

    static TokenMemoryPool& instance() {
        static TokenMemoryPool pool;
        return pool;
    }

    Ptr allocate() {
        Block* head = free_.load(std::memory_order_relaxed);
        do {
            if (!head) return Ptr(nullptr);  // 池耗尽
        } while (!free_.compare_exchange_weak(
            head, head->next,
            std::memory_order_acquire, std::memory_order_relaxed));

        T* obj = new (&head->data) T();  // placement new
        return Ptr(obj);
    }

private:
    TokenMemoryPool() {
        for (size_t i = 0; i + 1 < N; ++i)
            arena_[i].next = &arena_[i + 1];
        arena_[N - 1].next = nullptr;
        free_.store(&arena_[0], std::memory_order_release);
    }

    void recycle(T* p) {
        p->~T();
        Block* b = reinterpret_cast<Block*>(p);
        Block* head = free_.load(std::memory_order_relaxed);
        do {
            b->next = head;
        } while (!free_.compare_exchange_weak(
            head, b,
            std::memory_order_release, std::memory_order_relaxed));
    }

    Block arena_[N];
    std::atomic<Block*> free_{nullptr};
};

struct TokenCache {
    uint32_t token_id = 0;
    float logits[8] = {};
};

int main() {
    using Pool = TokenMemoryPool<TokenCache, 64>;

    {
        auto p = Pool::instance().allocate();
        if (p) {
            p->token_id = 42;
            std::cout << "Allocated Token ID: " << p->token_id << "\n";
        }
    }  // p 析构 -> Deleter -> Block 自动归还

    // 再分配一次, 验证回收成功
    auto p2 = Pool::instance().allocate();
    std::cout << "Re-allocated after recycle: " << (p2 ? "ok" : "fail") << "\n";
    return 0;
}
