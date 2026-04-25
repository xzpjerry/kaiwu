<div align="center">

# Kaiwu · 开物

**Auto-tuned local LLM serving: Kaiwu probes your hardware, model, KV cache, and context window so you get the fastest OpenAI-compatible endpoint your machine can actually sustain.**

**自动调优本地大模型：Kaiwu 探测你的硬件、模型、KV cache 和上下文窗口，给你一个机器能稳定跑出的最快 OpenAI 兼容端点。**

[English](#english) · [中文](#中文)

</div>

---

<a name="english"></a>
# Kaiwu

LM Studio and Ollama make models run. Kaiwu makes them run *well* — by measuring, not guessing.

It probes your GPU, reads the model architecture, benchmarks KV cache options, and walks the context window down from the model's native maximum until it finds the largest window your hardware can sustain at a useful speed. That config is cached. Second launch takes 2 seconds.

## Proof

### 30B MoE on 8GB GPU — the hard case

Model: Qwen3-30B-A3B · RTX 5060 Laptop 8GB · Windows 11

| | LM Studio | Kaiwu |
|---|---|---|
| Speed | 3 tok/s | **21 tok/s** |
| VRAM used | 7,549 MB (93%) | 2,603 MB (32%) |
| Config required | Manual | **None** |

LM Studio fills VRAM and saturates the GPU. Kaiwu detects the MoE architecture, keeps attention layers on GPU, routes expert layers through CPU — 7× faster, 65% less VRAM.

### 8B dense — everyday use

Model: Llama 3.1 8B Q5_K_M · RTX 5060 8GB

| | LM Studio | Kaiwu |
|---|---|---|
| Speed (8K ctx) | 46.5 tok/s | **51.7 tok/s** |
| Context window | 4–8K (default) | **64K (auto)** |

Same speed, 8× more context. Kaiwu calculates whether f16 KV cache fits in VRAM and uses it when it does — matching LM Studio's speed while running a much larger context window.

### Dual 4090 — high-end

Model: Qwen3.6-35B-A3B · 2× RTX 4090 24GB

- **115 tok/s** · **256K context** · fully automatic tensor split

## How It Works

```
kaiwu run Qwen3-30B-A3B
```

That's it. Kaiwu:

1. **Probes your hardware** — GPU model, VRAM, memory bandwidth, SM version, CPU cores, RAM
2. **Reads the model** — architecture, layer count, KV heads, native context limit, MoE structure
3. **Selects KV cache** — calculates f16 footprint; uses f16 if it fits, q8_0+q4_0 if not, iso3 for tight VRAM
4. **Runs warmup benchmark** — walks ctx from native max downward, stops where speed ≥ 20 tok/s
5. **Tunes parameters** — ubatch size, thread count, mlock — all measured, not guessed
6. **Caches the result** — next launch skips warmup entirely (2s startup)

On subsequent runs:

```
✓ Using last config  (64K ctx · 26.2 tok/s · 3 days ago)
```

## Installation

**Windows** (PowerShell):
```powershell
irm https://raw.githubusercontent.com/val1813/kaiwu/main/install.ps1 | iex
```

**Linux / macOS**:
```bash
curl -fsSL https://raw.githubusercontent.com/val1813/kaiwu/main/install.sh | sh
```

Or download manually from [Releases](https://github.com/val1813/kaiwu/releases).

## Quick Start

```bash
# Run a model (auto-downloads if needed)
kaiwu run Qwen3-30B-A3B

# Run a local GGUF file
kaiwu run /path/to/model.gguf

# Connect your IDE (Continue, Cursor, Claude Code)
# Point it to: http://localhost:11435/v1

# Check what's running
kaiwu status

# Stop
kaiwu stop
```

The API is OpenAI-compatible. Any tool that works with the OpenAI API works with Kaiwu.

## Advanced Usage

```bash
# Override context size
kaiwu run Qwen3-8B --ctx-size 12000

# Force re-tune (after hardware change)
kaiwu run Qwen3-8B --reset

# Fast start — skip warmup, use cached config only
kaiwu run Qwen3-8B --fast

# List available models
kaiwu list

# Inject IDE config automatically
kaiwu inject
```

## What Gets Auto-Tuned

| Parameter | How Kaiwu decides |
|---|---|
| Context length | Walks from model's native max down; stops where speed ≥ 20 tok/s |
| KV cache type | Calculates f16 footprint; uses f16 → q8_0+q4_0 → iso3 by VRAM fit |
| MoE expert placement | Detects `.ffn_.*_exps.` tensors; routes to CPU automatically |
| ubatch size | Benchmarks 128 vs 512; picks the faster one |
| Thread count | 2 for full-GPU, physical_cores/2 for MoE offload |
| mlock | Enabled when RAM headroom > 30% |
| GPU tensor split | Proportional to VRAM when multiple GPUs detected |

## Requirements

- **GPU**: NVIDIA (CUDA) — 4GB+ VRAM recommended
- **OS**: Windows 10/11, Linux (Ubuntu 20.04+)
- **RAM**: 8GB+ (16GB+ for 30B MoE models)
- **Model format**: GGUF

CPU-only inference is supported but not the focus.

## Commands

| Command | What it does |
|---|---|
| `run <model>` | Start a model. Downloads if needed. |
| `stop` | Stop the running model. |
| `status` | Show running model, speed, VRAM usage. |
| `list` | List available and downloaded models. |
| `probe` | Show detected hardware. |
| `inject` | Configure Continue/Cursor to use Kaiwu. |
| `version` | Show version. |

## Changelog

### v0.1.6 — MoE offload fix + direct path fix
- Fixed MoE offload warmup always OOM: replaced `-ot` regex with `--cpu-moe` (more reliable)
- `SelectKVCacheType` no longer guesses MoE GPU footprint — trusts `--fit on` for layer allocation
- Warmup timeout 60s → 180s for large MoE model loading
- Fixed `kaiwu run /path/to/model.gguf` silently downloading instead of using local file

### v0.1.5 — iso3 cache + bandwidth-aware tuning
- iso3 detection result cached to disk — same binary only detects once
- SM-aware timeout: SM<75 skipped, SM75-119 uses 15s, SM120+ uses 60s
- OOM suggestion copy is now dynamic — small models no longer wrongly told to switch to MoE
- Small models (<2GB) ubatch reduced 512→128, fixing `--kv-unified` pre-allocation OOM
- Memory bandwidth calculated from nvidia-smi XML (`bus_width × max_mem_clock × 2`)
- Low-bandwidth GPUs (<200 GB/s) only benchmark ubatch=128, saving 1-2 min warmup
- Full GPU bandwidth table (GTX 10/16/20/30/40/50 + datacenter V100/P100/H200)

### v0.1.4 — Blackwell architecture support
- Fixed iso3 detection timeout on RTX 50-series (SM120): 10s → 60s
- Root cause: CUDA 12.4 has no SM120 precompiled kernels; PTX JIT takes ~30s on first run
- Prints warning when SM120 detected: `⚠ RTX 50-series first launch requires JIT compilation (~30s)`

### v0.1.3 — hybrid architecture + APEX quantization
- APEX quantization presets: Quality (q8_0) / Balanced (q5_k_m) / Compact (q4_k_m)
- Hybrid architecture detection: auto-disables iso3 + enables `--swa-full` for DeltaNet/SSM models
- Direct GGUF path support: `kaiwu run /path/to/model.gguf`
- Flash Attention auto-enabled on SM75+
- NVLink auto-detection
- nvidia-smi XML parsing (replaces fragile CSV)
- Fixed multi-GPU VRAM calculation

### v0.1.2 — iso3 detection timing fix
- Moved iso3 detection to Preflight (before warmup)
- Root cause: warmup launched llama-server with iso3 flags before confirming support → all ctx probes failed, false OOM

### v0.1.1 — initial release
- Hardware probe: GPU (nvidia-smi), CPU, RAM
- Model matcher: VRAM-based quantization selection, full_gpu / moe_offload modes
- Warmup benchmark: binary search for max ctx at ≥20 tok/s, ubatch measurement
- Config cache: results saved to `~/.kaiwu/profiles/`, 2s second launch
- Bundled turboquant iso3 llama-server binary
- OpenAI-compatible API at `http://localhost:11435/v1`

---

<a name="中文"></a>
# 开物 (Kaiwu)

> *"开物成务，利用厚生"* — 明·宋应星《天工开物》

LM Studio 和 Ollama 让模型能跑。Kaiwu 让模型跑好——靠实测，不靠猜。

它探测你的 GPU、读取模型架构、测试 KV cache 选项，然后从模型的原生最大上下文往下走，找到你的硬件能以实用速度稳定跑出的最大窗口。结果缓存起来，第二次启动只需 2 秒。

## 数据说话

### 8GB 显卡跑 30B 模型——最难的场景

模型：Qwen3-30B-A3B · RTX 5060 笔记本 8GB · Windows 11

| | LM Studio | Kaiwu |
|---|---|---|
| 速度 | 3 tok/s | **21 tok/s** |
| 显存占用 | 7,549 MB（93%） | 2,603 MB（32%） |
| 需要手动配置 | 是 | **不需要** |

LM Studio 把显存塞满，GPU 跑满。Kaiwu 识别出 MoE 架构，只把 attention 层放 GPU，expert 层走 CPU——快 7 倍，省 65% 显存。

### 8B 模型——日常使用

模型：Llama 3.1 8B Q5_K_M · RTX 5060 8GB

| | LM Studio | Kaiwu |
|---|---|---|
| 速度（8K 上下文） | 46.5 tok/s | **51.7 tok/s** |
| 上下文窗口 | 4–8K（默认） | **64K（自动）** |

速度持平甚至更快，上下文多 8 倍。Kaiwu 计算 f16 KV cache 能不能装进显存，能装就用——速度匹配 LM Studio，同时跑更大的上下文。

### 双 4090——高端配置

模型：Qwen3.6-35B-A3B · 2× RTX 4090 24GB

- **115 tok/s** · **256K 上下文** · 自动多卡分配

## 工作原理

```
kaiwu run Qwen3-30B-A3B
```

就这一句。Kaiwu 会：

1. **探测硬件** — GPU 型号、显存、内存带宽、SM 版本、CPU 核数、内存
2. **读模型信息** — 架构、层数、KV heads、原生上下文限制、MoE 结构
3. **选 KV cache** — 计算 f16 占用；能装用 f16，不够降 q8_0+q4_0，显存极紧用 iso3
4. **跑 warmup 基准测试** — 从最大上下文往下探，找速度 ≥ 20 tok/s 的最大值
5. **调整参数** — ubatch 大小、线程数、mlock——全部实测，不靠猜
6. **缓存结果** — 下次启动跳过 warmup，2 秒就绪

第二次启动你会看到：

```
✓ 使用上次配置  (64K ctx · 26.2 tok/s · 3 天前)
```

## 安装

**Windows** (PowerShell):
```powershell
irm https://raw.githubusercontent.com/val1813/kaiwu/main/install.ps1 | iex
```

**Linux / macOS**:
```bash
curl -fsSL https://raw.githubusercontent.com/val1813/kaiwu/main/install.sh | sh
```

也可以从 [Releases](https://github.com/val1813/kaiwu/releases) 手动下载。

## 快速开始

```bash
# 运行模型（没有会自动下载）
kaiwu run Qwen3-30B-A3B

# 运行本地 GGUF 文件
kaiwu run /path/to/model.gguf

# 接入 IDE（Continue、Cursor、Claude Code）
# API 地址：http://localhost:11435/v1

# 查看运行状态
kaiwu status

# 停止
kaiwu stop
```

API 兼容 OpenAI 格式，任何支持 OpenAI API 的工具都可以直接用。

## 进阶用法

```bash
# 指定上下文大小
kaiwu run Qwen3-8B --ctx-size 12000

# 强制重新调参（换了硬件后）
kaiwu run Qwen3-8B --reset

# 快速启动——跳过 warmup，直接用缓存
kaiwu run Qwen3-8B --fast

# 列出可用模型
kaiwu list

# 自动配置 IDE
kaiwu inject
```

## 自动调整的参数

| 参数 | Kaiwu 怎么决定 |
|---|---|
| 上下文长度 | 从模型最大值往下探，找速度 ≥ 20 tok/s 的最大值 |
| KV cache 类型 | 计算 f16 占用；按显存依次选 f16 → q8_0+q4_0 → iso3 |
| MoE expert 位置 | 自动识别 `.ffn_.*_exps.` 张量，路由到 CPU |
| ubatch 大小 | 实测 128 vs 512，取快的 |
| 线程数 | 全 GPU 用 2，MoE offload 用物理核 /2 |
| mlock | 内存余量 > 30% 时自动开，防止模型被换出到磁盘 |
| 多卡分配 | 按显存比例自动切分 |

## 硬件要求

- **显卡**：NVIDIA（CUDA）——建议 4GB+ 显存
- **系统**：Windows 10/11，Linux（Ubuntu 20.04+）
- **内存**：8GB+（30B MoE 模型建议 16GB+）
- **模型格式**：GGUF

支持纯 CPU 推理，但不是主要使用场景。

## 命令列表

| 命令 | 说明 |
|---|---|
| `run <模型>` | 启动模型，没有会自动下载 |
| `stop` | 停止运行中的模型 |
| `status` | 显示当前模型、速度、显存占用 |
| `list` | 列出可用和已下载的模型 |
| `probe` | 显示检测到的硬件信息 |
| `inject` | 自动配置 Continue/Cursor 接入 Kaiwu |
| `version` | 显示版本号 |

## 版本历史

### v0.1.6 — MoE offload 修复 + 直接路径修复
- 修复 MoE offload warmup 全 OOM：用 `--cpu-moe` 替代 `-ot` 正则（更可靠）
- `SelectKVCacheType` 不再猜 MoE GPU 占用，直接信任 `--fit on` 处理层分配
- warmup 超时 60s → 180s（大 MoE 模型加载 ~13GB 需要更长时间）
- 修复 `kaiwu run /path/to/model.gguf` 实际走下载而非使用本地文件的 bug

### v0.1.5 — iso3 缓存 + 带宽感知调参
- iso3 检测结果缓存到磁盘，同一 binary 只检测一次
- SM 版本感知超时：SM<75 直接跳过，SM75-119 用 15s，SM120+ 用 60s
- OOM 建议文案动态化：小模型 OOM 不再错误建议换 MoE
- 小模型（<2GB）ubatch 从 512 降到 128，修复 `--kv-unified` 预分配 OOM
- 带宽从 nvidia-smi XML 精确计算（`bus_width × max_mem_clock × 2`）
- 低带宽卡（<200 GB/s）warmup 只测 ubatch=128，减少 1-2 分钟等待
- 完整 GPU 带宽枚举表（GTX 10/16/20/30/40/50 + 数据中心）

### v0.1.4 — Blackwell 架构支持
- 修复 RTX 50 系（SM120）iso3 检测超时：10s → 60s
- 根因：CUDA 12.4 无 SM120 预编译 kernel，PTX JIT 编译需 ~30s
- 检测到 SM120 时打印提示：`⚠ RTX 50 系首次启动需要 JIT 编译 (~30s)`

### v0.1.3 — 混合架构支持 + APEX 量化
- APEX 量化三档预设：Quality (q8_0) / Balanced (q5_k_m) / Compact (q4_k_m)
- 混合架构动态检测：iso3 自动禁用 + `--swa-full` 补偿（DeltaNet/SSM 架构）
- 直接 GGUF 路径支持：`kaiwu run /path/to/model.gguf`
- Flash Attention 自动启用（SM75+）
- NVLink 自动检测
- nvidia-smi XML 解析（替代脆弱的 CSV 解析）
- 修复多卡 VRAM 计算错误

### v0.1.2 — iso3 检测时机修复
- iso3 检测移到 warmup 之前（Preflight 阶段）
- 根因：warmup 用 iso3 参数启动 llama-server，但 binary 不支持 → 所有 ctx 探测失败，误报 OOM

### v0.1.1 — 初始版本
- 硬件探测：GPU（nvidia-smi）、CPU、内存
- 模型匹配：基于 VRAM 选量化，full_gpu / moe_offload 两种模式
- Warmup 基准测试：二分探测最大 ctx，ubatch 实测
- 配置缓存：结果保存到 `~/.kaiwu/profiles/`，第二次启动 2 秒
- 内置 turboquant iso3 llama-server binary
- OpenAI 兼容 API：`http://localhost:11435/v1`

---

## For Developers / 贡献者

Build from source (requires Go 1.22+):

```bash
git clone https://github.com/val1813/kaiwu.git
cd kaiwu
make build-windows   # or build-linux
```

---

<div align="center">

Built on [llama.cpp](https://github.com/ggerganov/llama.cpp) · by [llmbbs.ai](https://llmbbs.ai)

</div>
