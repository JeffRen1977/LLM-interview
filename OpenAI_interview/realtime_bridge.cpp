// 用户态 AI Runtime 软硬交互 —— 实时线程 + CPU 亲和度 + mmap 零拷贝 参考实现
//
// 编译:
//   Linux : g++ -std=c++17 -O2 -pthread realtime_bridge.cpp -o realtime_bridge
//   macOS : clang++ -std=c++17 -O2 realtime_bridge.cpp -o realtime_bridge
// 运行:
//   Linux 实时优先级/绑核需 root 或 CAP_SYS_NICE: sudo ./realtime_bridge
//   非 Linux 平台会自动降级(仅打印说明), 仍可编译运行, 便于本地演示。
//
// 设计要点:
//   1. 把 Event Loop 线程绑定到隔离核心, 避免 Core-to-Core 迁移导致 L1/L2 失效。
//   2. 用 SCHED_FIFO + 高优先级, 确保不被后台任务(Wi-Fi 扫描/日志)抢占。
//   3. 用 mmap 把内核/设备缓冲区映射进用户态, 实现零拷贝读写。
//   4. Linux 专属 API 用 #ifdef __linux__ 隔离, 其他平台优雅降级。

#include <cstring>
#include <iostream>

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#if defined(__linux__)
#include <pthread.h>
#include <sched.h>
#endif

// ─────────────────────────────────────────────────────────────
// 1. 设置当前线程的 CPU 亲和度 + 硬实时优先级 (SCHED_FIFO)
// ─────────────────────────────────────────────────────────────
bool configure_realtime_thread(int cpu_core, int rt_priority) {
#if defined(__linux__)
    // 绑核: 避免 Core-to-Core 迁移导致 L1/L2 Cache Invalidation
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(cpu_core, &cpuset);

    pthread_t self = pthread_self();
    if (pthread_setaffinity_np(self, sizeof(cpu_set_t), &cpuset) != 0) {
        std::cerr << "Failed to set CPU affinity to Core " << cpu_core << ": "
                  << std::strerror(errno) << "\n";
        return false;
    }

    // 实时调度策略 SCHED_FIFO, 优先级范围通常 1-99
    struct sched_param param{};
    param.sched_priority = rt_priority;
    if (pthread_setschedparam(self, SCHED_FIFO, &param) != 0) {
        std::cerr << "Failed to set SCHED_FIFO (need root / CAP_SYS_NICE?): "
                  << std::strerror(errno) << "\n";
        return false;
    }
    std::cout << "[RT] Bound to Core " << cpu_core << ", SCHED_FIFO prio "
              << rt_priority << "\n";
    return true;
#else
    // 非 Linux 平台 (如 macOS): 无 SCHED_FIFO / cpu_set_t, 优雅降级
    std::cout << "[RT] (non-Linux) skip affinity Core " << cpu_core
              << " / SCHED_FIFO prio " << rt_priority
              << " — 真实嵌入式 Linux 上才生效\n";
    return false;
#endif
}

// ─────────────────────────────────────────────────────────────
// 2. mmap 零拷贝映射内核/设备缓冲区 (RAII 管理 fd 与映射)
// ─────────────────────────────────────────────────────────────
class HardwareBufferBridge {
public:
    explicit HardwareBufferBridge(const char* device_path) {
        fd_ = ::open(device_path, O_RDWR);
        if (fd_ < 0) {
            std::cerr << "Failed to open " << device_path << ": "
                      << std::strerror(errno) << "\n";
        }
    }

    ~HardwareBufferBridge() {
        if (mapped_ && mapped_ != MAP_FAILED && size_ > 0) {
            ::munmap(mapped_, size_);
        }
        if (fd_ >= 0) ::close(fd_);
    }

    HardwareBufferBridge(const HardwareBufferBridge&) = delete;
    HardwareBufferBridge& operator=(const HardwareBufferBridge&) = delete;

    // 把设备/内核缓冲区映射到用户态, 返回可直接读写的指针 (零拷贝)
    uint8_t* map(size_t size, off_t offset = 0) {
        if (fd_ < 0) return nullptr;
        size_ = size;
        // MAP_SHARED: 改动对内核/其他映射者可见 (设备缓冲区语义)
        mapped_ = static_cast<uint8_t*>(
            ::mmap(nullptr, size_, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, offset));
        if (mapped_ == MAP_FAILED) {
            std::cerr << "mmap failed: " << std::strerror(errno) << "\n";
            mapped_ = nullptr;
        }
        return mapped_;
    }

    bool valid() const { return fd_ >= 0; }

private:
    int fd_ = -1;
    uint8_t* mapped_ = nullptr;
    size_t size_ = 0;
};

// ─────────────────────────────────────────────────────────────
// 3. 演示: 匿名 mmap 模拟"内核缓冲区"的零拷贝读写
//    (真实场景把 /dev/video0 等设备 fd 传入 HardwareBufferBridge)
// ─────────────────────────────────────────────────────────────
int main() {
    // 尝试把主线程提升为实时线程 (本地无权限会降级, 不影响演示)
    configure_realtime_thread(/*cpu_core=*/0, /*rt_priority=*/90);

    // 用匿名映射模拟一块 "内核/DMA 缓冲区"
    constexpr size_t kBufSize = 4096;
    uint8_t* buf = static_cast<uint8_t*>(
        ::mmap(nullptr, kBufSize, PROT_READ | PROT_WRITE,
               MAP_ANONYMOUS | MAP_PRIVATE, -1, 0));
    if (buf == MAP_FAILED) {
        std::cerr << "anonymous mmap failed: " << std::strerror(errno) << "\n";
        return 1;
    }

    // 零拷贝: 直接通过指针写入/读出, 无 read/write 系统调用与额外拷贝
    std::memset(buf, 0, kBufSize);
    buf[0] = 0xAB;
    buf[kBufSize - 1] = 0xCD;

    std::cout << "mmap buffer @ " << static_cast<void*>(buf)
              << ", size " << kBufSize << "\n";
    std::cout << "buf[0]=0x" << std::hex << static_cast<int>(buf[0])
              << ", buf[last]=0x" << static_cast<int>(buf[kBufSize - 1])
              << std::dec << "\n";

    ::munmap(buf, kBufSize);
    std::cout << "Done. (真实设备: 把 /dev/video0 fd 交给 HardwareBufferBridge::map)\n";
    return 0;
}
