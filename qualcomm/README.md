# Qualcomm 面试技能 · 动手实验室

> 对齐 [`document/qualcomm_staff_ai_software_tools_prep.md`](../document/qualcomm_staff_ai_software_tools_prep.md)  
> **目标**：在无真机 / 无 QAIRT SDK 的情况下，用可运行代码建立 **PyTorch → ONNX → ORT → 量化 → 图划分 → 端侧 LLM** 的「实际做过」叙事。

---

## 快速开始

```bash
# 仓库根目录
pip install -r requirements.txt
pip install -r qualcomm/requirements.txt

# 跑全部实验
python3 qualcomm/run_all.py

# 跑单个实验
python3 qualcomm/run_all.py --lab 01
python3 qualcomm/lab_03_quantization_ptq.py
```

**C++（可选，对应 JD Python + C/C++）**：

```bash
cd qualcomm/cpp && cmake -B build && cmake --build build && ./build/qnn_stub
```

---

## 实验地图

| Lab | 文件 | 练什么 | 对应 prep |
|-----|------|--------|-----------|
| **01** | `lab_01_onnx_export.py` | `torch.onnx.export`、checker、PyTorch vs ORT golden | §3.4, §3.6, §10.5 |
| **02** | `lab_02_graph_transform.py` | constant fold、MatMul+Add+Relu 融合（compiler pass 思维） | §3.4, §10.5 |
| **03** | `lab_03_quantization_ptq.py` | 对称量化、per-tensor/channel、动/静态 PTQ、QDQ | §3.1 |
| **04** | `lab_04_ort_runtime.py` | `InferenceSession`、EP 链、QNN EP fallback 模式 | §3.4, §10.5 |
| **05** | `lab_05_graph_partition_delegate.py` | HTP vs CPU 子图划分（delegate 核心） | §3.4, §10.5 |
| **06** | `lab_06_llm_prefill_decode.py` | TTFT/TPOT、KV 显存、端侧内存预算 | §3.2 |
| **07** | `lab_07_moe_routing.py` | top-k router、expert dispatch、端侧难点 | §11 |
| **08** | `lab_08_lora_merge_deploy.py` | `W' = W + BA`、merge vs sidecar | §3.5 |
| **09** | `lab_09_debug_golden_pipeline.py` | 跨层 golden、量化误差定位 | §3.9, §10.7 |
| **10** | `lab_10_e2e_pipeline.py` | **串起全流程** — 面试 30 秒故事 | §1 心智模型 |
| **C++** | `cpp/qnn_runtime_stub.cpp` | context load → graph execute（QNN API 形状） | §3.7, §10 |

---

## 与真实 QAIRT 栈的关系

```
本仓库 labs（Host，无需 Snapdragon）
    PyTorch → ONNX → ORT CPU → 自研 pass / partition 模拟

真机路径（需 Qualcomm Software Center SDK）
    PyTorch/ONNX → QNN converter → quantize → compile → context.bin
    → Genie / ORT QNNExecutionProvider → HTP
```

| 本 labs 模拟 | 真 SDK 对应 |
|-------------|-------------|
| Lab 02 graph fuse | QNN / ORT graph optimizer |
| Lab 03 PTQ | AIMET、QNN quantizer |
| Lab 05 partition | QNN EP / ExecuTorch QNN delegate |
| Lab 04 ORT | ONNX Runtime + `QNNExecutionProvider` |
| `cpp/qnn_stub` | `QnnContext_create`, `QnnGraph_execute` |

---

## 推荐学习顺序（2 周）

### Week 1 — 图与运行时

| 天 | Lab | 产出（写下来能讲） |
|----|-----|-------------------|
| 1 | 01 | export 踩坑列表 + max diff 截图 |
| 2 | 02, 05 | 画一张 partition 图（HTP/CPU） |
| 3 | 03 | per-channel vs per-tensor 误差数字 |
| 4 | 04 | 本机 `get_available_providers()` 输出 |
| 5 | 09 | 一套 debug 顺序（golden → quant → layout） |

### Week 2 — GenAI 与端到端

| 天 | Lab | 产出 |
|----|-----|------|
| 1 | 06 | 手算 3B 模型 KV + INT4 权重是否 fit 12GB |
| 2 | 07, 08 | MoE 端侧难点 + LoRA merge 一句话 |
| 3 | 10 | **完整 pipeline 口述**（配合 run_all 输出） |
| 4 | cpp | 编译跑通 stub，读 QNN C sample 对照 |
| 5 | — | 若有 SDK：AI Hub + `genie-t2t-run`（prep §10.6） |

---

## 面试怎么说（模板）

> 我在本地搭了一条 **Qualcomm 风格的部署链**：用 PyTorch 导出 ONNX，做了 **fold/fuse** 图优化，用 **ORT** 做 golden 对比，实现了 **per-channel INT8** 和 **HTP/CPU partition** 模拟；LLM 侧估了 **Prefill/Decode** 和 **KV 显存**；LoRA **merge** 后单图部署；并有一套 **逐层 diff** 的 debug 流程。真机上对应 **QNN compile + Genie + QNN EP**，我已在 SDK 文档里对齐了 API 流程。

---

## 目录结构

```
qualcomm/
  README.md                 ← 本文件
  requirements.txt
  run_all.py
  common/utils.py
  lab_01_onnx_export.py
  ...
  lab_10_e2e_pipeline.py
  cpp/
    qnn_runtime_stub.cpp
    CMakeLists.txt
```

---

## 相关仓库章节

| 主题 | 文档 | 代码 |
|------|------|------|
| 量化理论 | [chapter_07](../document/chapter_07_model_quantization.md) | `basic/chapter_07_*.py` |
| LLM 推理 | [chapter_08](../document/chapter_08_inference_pipeline.md) | `basic/chapter_08_*.py` |
| 算子融合 | [chapter_09](../document/chapter_09_flash_attention_operator_fusion.md) | `basic/chapter_09_*.py` |
| MoE | prep §11 | `basic/chapter_16_moe_inference.py` |

---

*有 SDK + Snapdragon 设备后，在 Lab 10 叙事末尾加一句：「我已在真机上用 QNN EP / Genie 跑通 binary」。*
