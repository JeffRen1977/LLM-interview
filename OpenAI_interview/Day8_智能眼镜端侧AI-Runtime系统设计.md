# Day 8 系统设计：智能眼镜端侧 AI Runtime

> 场景：为智能眼镜（Smart Glasses）这种超低功耗、高实时性物理设备设计 AI Runtime。
> 核心理念：云端设计范式失效，必须在 **算力约束、内存墙（Memory Wall）、电池功耗（Thermal & Battery）** 三个物理边界内做极限设计。
> 白板答题主线：先定约束 → 分层架构 → 省电（级联唤醒）→ 低延迟（零拷贝/量化/投机采样）→ 端云路由 → Q&A。

---

## 目录
1. [约束与物理边界定义](#1-约束与物理边界定义-requirements--constraints)
2. [软件运行架构设计](#2-软件运行架构设计-software-architecture)
3. [极致省电：Always-On 状态机与中断级联唤醒](#3-极致省电always-on-状态机与中断驱动机制)
4. [极致低延迟：零拷贝、量化、投机采样](#4-极致低延迟基于-executorch-的内存零拷贝与模型压缩)
5. [端-云协同动态路由决策器](#5-端-云协同动态路由决策器-hybrid-decision-router)
6. [面试官追问 Q&A](#6-面试官高频追问模拟-qa-strategy)
7. [口头表达速记卡](#7-口头表达速记卡闭眼默背)

---

## 1. 约束与物理边界定义 (Requirements & Constraints)

白板第一步：主动向面试官明确硬件物理极限。

- **功耗与散热（Thermal/Power）**：整机功耗预算 **1.5W–2.5W**（防止眼镜发烫烫伤皮肤）；电池仅 **400mAh–600mAh**（约 2–3 小时持续多模态交互）。
- **内存（RAM）**：边缘 SOC（如骁龙 AR1 / AR2 Gen 1）通常只有 **2GB–4GB LPDDR5**，且 **CPU / GPU / NPU 共享**这块物理内存。
- **低延迟指标（Latency）**：
  - **语音唤醒（Wake Word）**：< 100ms（完全本地）。
  - **视觉多模态问答（VLA / VLM）**：首字延迟 **TTFT 500ms–1s** 以内。

---

## 2. 软件运行架构设计 (Software Architecture)

采用 **端-云协同的分层混合（Hybrid）架构**。

```
 ┌─────────────────────────────────────────────────────────────────────────┐
 │                      智能眼镜多模态 Agent 运行时系统                       │
 └─────────────────────────────────────────────────────────────────────────┘

   [ 传感器采集层 ]             [ 本地极低功耗处理层 (Always-On Subsystem) ]

   ┌─────────────┐             ┌────────────────────────────────┐
   │ 麦克风 (I2S)├────────────►│ 1. 硬件 VAD / 唤醒检测 (DSP)   │
   └─────────────┘             └────────────────┬───────────────┘
                                                │
   ┌─────────────┐             ┌────────────────▼───────────────┐
   │ 摄像头 (MIPI)├───────────►│ 2. 视觉变化检测 (Motion Detect)│
   └─────────────┘             └────────────────┬───────────────┘
                                                │
                                                │ (触发唤醒 / 状态机改变)
                                                ▼
                               [ 端侧 AI 推理层 (Main SOC / NPU / GPU) ]

                               ┌────────────────────────────────┐
                               │ 3. 边缘端 Agent 调度器 (C++)   │
                               │    - 零拷贝 环形缓存队列        │
                               │    - 动态 KV Cache 缓存管理器  │
                               └────────────────┬───────────────┘
                                                │
                               ┌────────────────▼───────────────┐
                               │ 4. 本地端模型链 (ExecuTorch)   │
                               │    - Whisper-tiny (INT8)       │
                               │    - Mini-VLM / LLaVA (INT4)   │
                               └────────────────┬───────────────┘
                                                │
                                                ▼ (低带宽 / 超长上下文 / 复杂工具)
                               [ 混合网络路由层 (Cloud Offloading) ]

                               ┌────────────────────────────────┐
                               │ 5. 远端高性能推理 (云端 LMM)    │
                               └────────────────────────────────┘
```

---

## 3. 极致省电：Always-On 状态机与中断驱动机制

不让主 SOC 一直高功耗活跃，底层设计**多层级级联唤醒机制**：

### 第一层：超低功耗音频 DSP（Always-On Audio）
- 主 SOC 处于深度休眠（Deep Sleep）。
- 音频经 I2S/PDM 持续流向一颗独立的、功耗 **< 10mW 的微型 DSP**。
- DSP 上跑极轻量 **VAD + 本地唤醒词模型**，只有检测到有效唤醒词才发**硬件中断（IRQ）**唤醒主 SOC 全功率工作。

### 第二层：超低功耗摄像头（Always-On Vision）
- 静止状态下相机不采高清，而是低分辨率（如 320×240 @ 5fps）。
- 超低功耗协处理器做**运动变化检测（Motion Detection）**；只有画面显著变化，或用户主动触发（双击镜腿 → IMU 中断）才拉高 MIPI 时钟、唤醒主传感器采高清帧。

> 关键词：级联唤醒（cascaded wake）、功耗分层、IRQ 触发状态机切换。

---

## 4. 极致低延迟：基于 ExecuTorch 的内存零拷贝与模型压缩

### A. 传感器数据到推理引擎的零拷贝（Zero-Copy）
- **技术**：Linux **`DMA-BUF`**。相机驱动把图像采到物理连续内存（CMA），不经用户态 `memcpy`，直接把物理内存的硬件 fd 以指针引用传给 NPU 推理引擎（**ExecuTorch** 的 NPU Backend Delegate）。
- **效果**：1080P 多模态输入图像搬移开销从 30–50ms 降到 **≈ 0ms**。

### B. 端侧模型量化与推理加速
- **量化策略（Quantization）**：
  - 音频 ASR（Whisper-tiny）：**INT8**，语义不失真 + 契合 NPU 8-bit 整型算子吞吐。
  - 多模态视觉小模型（LLaVA-3B / Mini-VLM）：**INT4（Group-wise）混合量化**——权重 4-bit，关键激活层用 FP16 混合精度，维持精度。
- **KV Cache 静态预分配与回收**：
  - 端侧**绝不用动态 `new`/`malloc`** 增长 KV Cache（堆碎片 + 频繁 OOM）。
  - **手段**：分块（Block-based）静态预分配，初始化阶段一次性占满 NPU 物理显存；无锁环形队列管理 Block + **LRU 回收策略**，生成新 Token 时回收前序无用上下文 Block。

### C. 投机采样（Speculative Decoding）
- 利用异构算力：轻量小模型（100M–200M）在 CPU/GPU 高频跑草稿 Token；大模型（3B VLM）在 NPU 上**一次性并行验证**。
- 端侧解码吞吐提升 **1.5x–2x**。

---

## 5. 端-云协同动态路由决策器 (Hybrid Decision Router)

不是所有请求都适合本地，也不是都适合上云（弱网/离线）。引入**轻量启发式决策路由层**：

```cpp
enum class InferenceRoute {
    LOCAL_NPU,   // 纯本地: 查时间、拍照、简单实体识别
    CLOUD_LMM    // 云端: 长翻译、复杂逻辑推理
};

InferenceRoute make_routing_decision(const std::string& intent, double current_rtt) {
    // 1. 网络检测: 延迟过高或断网 -> 强制本地
    if (current_rtt > 300.0) return InferenceRoute::LOCAL_NPU;
    // 2. 意图分析: 复杂/长任务 -> 上云
    if (intent == "complex_reasoning" || intent == "long_translation")
        return InferenceRoute::CLOUD_LMM;
    return InferenceRoute::LOCAL_NPU;
}
```

> 完整可编译 demo（路由决策 + KV Cache LRU 分块回收）见 [`smart_glasses_runtime.cpp`](./smart_glasses_runtime.cpp)。

---

## 6. 面试官高频追问模拟 (Q&A Strategy)

### 追问 1：多模态（摄像头 + 麦克风）输入下，如何解决多传感器时间戳漂移？
**答**：底层采用**硬件级同步（Hardware Time-Stamping）**。用 SOC 硬件 PWM 信号作相机的垂直同步（V-Sync），麦克风采集驱动使用相同系统参考时钟（如 `CLOCK_MONOTONIC`）。数据到达 Runtime 的无锁 SPSC 队列后，用单线程事件循环将二者对齐到同一物理时间窗口（呼应 Day 1–2 的滑动时间窗口 + 二分查找）。

### 追问 2：前向传播中 NPU 突然被高优先级任务（AR 渲染几何解算）占用怎么办？
**答**：利用异构计算（Heterogeneous Computing）。设计时配置好「热备份路由」：Runtime 监控到 NPU 被独占 / 负载过高，就把已量化、且支持 GPU 编译后端的模型瞬间卸载分流到 **GPU / CPU** 并行跑，确保听觉/视觉交互绝不卡顿。

---

## 7. 口头表达速记卡（闭眼默背）

- **先定三边界**：功耗 1.5–2.5W、RAM 2–4GB 共享、TTFT 500ms–1s。
- **省电级联唤醒**：DSP(<10mW) VAD/唤醒词 → 协处理器 Motion Detect → IRQ 唤醒主 SOC。
- **零拷贝**：DMA-BUF + CMA，把物理内存 fd 直接交 ExecuTorch NPU delegate，1080P 搬移 ≈0ms。
- **量化**：ASR INT8；VLM INT4 group-wise + 关键激活 FP16 混合精度。
- **KV Cache**：静态预分配分块 + 无锁环形队列 + LRU 回收，杜绝动态分配。
- **投机采样**：小模型草稿(CPU/GPU) + 大模型并行验证(NPU)，1.5–2x。
- **端云路由**：RTT>300ms 强制本地；复杂/长任务上云；其余本地。
- **时间戳同步**：硬件 V-Sync + `CLOCK_MONOTONIC` + 单线程事件循环对齐窗口。
- **NPU 抢占**：热备份路由，分流到 GPU/CPU。
