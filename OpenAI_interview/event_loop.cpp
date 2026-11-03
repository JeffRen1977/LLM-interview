// 嵌入式多传感器异步事件循环 —— 无锁 SPSC + 零拷贝意向 参考实现
//
// 编译: g++ -std=c++17 -O2 -pthread event_loop.cpp -o event_loop
// 运行: ./event_loop
//
// 设计要点:
//   1. 每个传感器一条独立的 SPSC 无锁环形队列 (单生产者单消费者)。
//   2. Acquire-Release 语义完成生产者/消费者握手, 不使用 seq_cst。
//   3. head/tail 各自 alignas(64) 隔离 Cache Line, 避免 False Sharing。
//   4. 事件循环在有数据时 Active Polling, 空闲时用条件变量休眠省电。

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <iostream>
#include <mutex>
#include <thread>

// ─────────────────────────────────────────────────────────────
// 单生产者单消费者 (SPSC) 无锁环形缓冲区
// ─────────────────────────────────────────────────────────────
template <typename T, size_t Capacity>
class LockFreeSPSCQueue {
public:
    static_assert((Capacity & (Capacity - 1)) == 0,
                  "Capacity must be a power of 2");

    // 生产者调用: 只写 tail, 用 release 暴露数据
    bool push(const T& item) {
        const size_t current_tail = tail_.load(std::memory_order_relaxed);
        const size_t current_head = head_.load(std::memory_order_acquire);

        if ((current_tail - current_head) == Capacity) {
            return false;  // 队列满
        }

        buffer_[current_tail & (Capacity - 1)] = item;
        // Release: 先写 buffer, 再暴露 tail, 保证消费者看到指针时数据已就绪
        tail_.store(current_tail + 1, std::memory_order_release);
        return true;
    }

    // 消费者调用: 只写 head, 用 acquire 读取最新数据
    bool pop(T& item) {
        const size_t current_head = head_.load(std::memory_order_relaxed);
        const size_t current_tail = tail_.load(std::memory_order_acquire);

        if (current_head == current_tail) {
            return false;  // 队列空
        }

        item = buffer_[current_head & (Capacity - 1)];
        // Release: 数据取出后再更新 head, 释放槽位给生产者
        head_.store(current_head + 1, std::memory_order_release);
        return true;
    }

private:
    T buffer_[Capacity];
    // alignas(64): 隔离到不同 Cache Line, 消除伪共享 (False Sharing)
    alignas(64) std::atomic<size_t> head_{0};
    alignas(64) std::atomic<size_t> tail_{0};
};

// ─────────────────────────────────────────────────────────────
// 抽象传感器数据事件 (零拷贝: 只传指针/物理内存句柄)
// ─────────────────────────────────────────────────────────────
struct SensorEvent {
    enum class Type { IMU, Camera, Audio };
    Type type{};
    uint64_t timestamp{};
    void* raw_data_ptr{nullptr};
};

static uint64_t now_ns() {
    using namespace std::chrono;
    return static_cast<uint64_t>(
        duration_cast<nanoseconds>(steady_clock::now().time_since_epoch())
            .count());
}

// ─────────────────────────────────────────────────────────────
// 事件循环核心 (单线程 Reactor)
// ─────────────────────────────────────────────────────────────
class EventLoop {
public:
    EventLoop() : running_(true) {}

    // 由各传感器采集线程异步调用
    void dispatch_event(const SensorEvent& event) {
        bool ok = true;
        switch (event.type) {
            case SensorEvent::Type::IMU:
                ok = imu_queue_.push(event);
                break;
            case SensorEvent::Type::Camera:
                ok = camera_queue_.push(event);
                break;
            case SensorEvent::Type::Audio:
                ok = audio_queue_.push(event);
                break;
        }
        if (!ok) {
            // 队列满: 实时系统里选择 Drop, 绝不在 I/O 线程动态分配内存
            dropped_.fetch_add(1, std::memory_order_relaxed);
        }
        wakeup_event_loop();  // 若循环在休眠则唤醒
    }

    void run() {
        SensorEvent event;
        while (running_.load(std::memory_order_relaxed)) {
            bool processed_any = false;

            // 1. 无锁轮询高优先级 IMU (高频优先)
            while (imu_queue_.pop(event)) {
                process(event);
                processed_any = true;
            }
            // 2. 轮询低频传感器
            while (camera_queue_.pop(event)) {
                process(event);
                processed_any = true;
            }
            while (audio_queue_.pop(event)) {
                process(event);
                processed_any = true;
            }

            // 3. 低功耗策略: 无数据时休眠, 由 dispatch/stop 唤醒
            if (!processed_any) {
                wait_for_hardware_interrupt();
            }
        }
    }

    void stop() {
        running_.store(false, std::memory_order_relaxed);
        wakeup_event_loop();
    }

    uint64_t processed() const {
        return processed_.load(std::memory_order_relaxed);
    }
    uint64_t dropped() const {
        return dropped_.load(std::memory_order_relaxed);
    }

private:
    void process(const SensorEvent& e) {
        // 真实系统: 快速分发给端侧 Agent 推理模块 / 时间戳对齐
        processed_.fetch_add(1, std::memory_order_relaxed);
        (void)e;
    }

    // 用条件变量模拟 epoll_wait / futex 休眠 (可移植的演示实现)
    void wait_for_hardware_interrupt() {
        std::unique_lock<std::mutex> lock(wake_mtx_);
        cv_.wait_for(lock, std::chrono::milliseconds(2), [this] {
            return wake_pending_ || !running_.load(std::memory_order_relaxed);
        });
        wake_pending_ = false;  // 防虚假唤醒: predicate + 清标志
    }

    void wakeup_event_loop() {
        {
            std::lock_guard<std::mutex> lock(wake_mtx_);
            wake_pending_ = true;
        }
        cv_.notify_one();
    }

    std::atomic<bool> running_;
    std::atomic<uint64_t> processed_{0};
    std::atomic<uint64_t> dropped_{0};

    // 每个传感器一条独立 SPSC 队列
    LockFreeSPSCQueue<SensorEvent, 1024> imu_queue_;
    LockFreeSPSCQueue<SensorEvent, 256> camera_queue_;
    LockFreeSPSCQueue<SensorEvent, 512> audio_queue_;

    // 休眠/唤醒机制
    std::mutex wake_mtx_;
    std::condition_variable cv_;
    bool wake_pending_{false};
};

// ─────────────────────────────────────────────────────────────
// 演示: 3 个传感器线程 + 1 个事件循环线程
// ─────────────────────────────────────────────────────────────
int main() {
    EventLoop loop;
    std::thread loop_thread([&loop] { loop.run(); });

    auto producer = [&loop](SensorEvent::Type type, int hz, int count) {
        const auto period = std::chrono::nanoseconds(1'000'000'000 / hz);
        for (int i = 0; i < count; ++i) {
            SensorEvent e;
            e.type = type;
            e.timestamp = now_ns();
            e.raw_data_ptr = nullptr;  // 零拷贝: 真实场景传物理页指针
            loop.dispatch_event(e);
            std::this_thread::sleep_for(period);
        }
    };

    // IMU 200Hz, Camera 30Hz, Audio(I2C 类) 10Hz, 各跑 ~0.5s
    std::thread imu(producer, SensorEvent::Type::IMU, 200, 100);
    std::thread cam(producer, SensorEvent::Type::Camera, 30, 15);
    std::thread aud(producer, SensorEvent::Type::Audio, 10, 5);

    imu.join();
    cam.join();
    aud.join();

    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    loop.stop();
    loop_thread.join();

    std::cout << "Processed events: " << loop.processed() << "\n";
    std::cout << "Dropped events:   " << loop.dropped() << "\n";
    std::cout << "Expected total:   " << (100 + 15 + 5) << "\n";
    return 0;
}
