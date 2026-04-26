package engine

import (
	"fmt"
	"net"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/val1813/kaiwu/internal/config"
	"github.com/val1813/kaiwu/internal/hardware"
	"github.com/val1813/kaiwu/internal/model"
)

// RunningEngine represents a running llama-server instance
type RunningEngine struct {
	PID        int
	Port       int
	ModelID    string
	BinaryPath string
	LogPath    string
	CtxSize    int      // 实际使用的上下文大小
	logFile    *os.File // 保持日志文件句柄，进程退出时再关
}

// Start starts llama-server with probe-and-retry strategy.
// 探测式启动：尝试最优 ctx → OOM 就减半重试 → 最多 3 次。
// 注意：iso3 检测已在 main.go [4/6] Preflight 阶段完成，profile.HasIsoQuant 已更新。
func Start(profile *model.DeployProfile, binaryPath, modelPath string, hw *hardware.HardwareProbe) (*RunningEngine, error) {
	ctxSize := IdealStartCtx(profile, hw)

	for attempt := 0; attempt < 3; attempt++ {
		if attempt > 0 {
			fmt.Printf("      ⚠️  显存不足，降低上下文至 %dK 重试...\n", ctxSize/1024)
		}

		eng, err := startOnce(profile, binaryPath, modelPath, hw, ctxSize)
		if err == nil {
			eng.CtxSize = ctxSize
			return eng, nil
		}

		// 判断是否 OOM（进程启动后很快退出 = 大概率 OOM）
		if !isLikelyOOM(err) {
			return nil, err
		}

		// 清理失败的进程
		Stop()

		// ctx 减半
		ctxSize = ctxSize / 2
		if ctxSize < 4096 {
			ctxSize = 4096
		}

		// 已经是最小了还失败 → 给出详细的 VRAM 分析
		if ctxSize == 4096 && attempt > 0 {
			return nil, buildOOMError(profile, hw, attempt+1)
		}
	}

	return nil, fmt.Errorf("3 次启动均失败，建议选择更小的模型")
}

// StartWithArgs starts llama-server with pre-optimized args from warmup.
func StartWithArgs(profile *model.DeployProfile, binaryPath, modelPath string, hw *hardware.HardwareProbe, optimizedArgs []string) (*RunningEngine, error) {
	if len(optimizedArgs) == 0 {
		return Start(profile, binaryPath, modelPath, hw)
	}

	cfg, err := config.Load()
	if err != nil {
		return nil, err
	}

	actualPort := findFreePort(cfg.LlamaPort)
	if actualPort != cfg.LlamaPort {
		fmt.Printf("Port %d in use, using %d instead\n", cfg.LlamaPort, actualPort)
	}

	// Patch port in optimized args
	args := make([]string, len(optimizedArgs))
	copy(args, optimizedArgs)
	for i, a := range args {
		if a == "--port" && i+1 < len(args) {
			args[i+1] = strconv.Itoa(actualPort)
			break
		}
	}

	return launchProcess(profile, binaryPath, args, actualPort)
}

// startOnce 单次启动尝试
func startOnce(profile *model.DeployProfile, binaryPath, modelPath string, hw *hardware.HardwareProbe, ctxSize int) (*RunningEngine, error) {
	cfg, err := config.Load()
	if err != nil {
		return nil, err
	}

	actualPort := findFreePort(cfg.LlamaPort)
	if actualPort != cfg.LlamaPort {
		fmt.Printf("Port %d in use, using %d instead\n", cfg.LlamaPort, actualPort)
	}

	args := buildArgs(profile, modelPath, actualPort, hw, ctxSize)
	return launchProcess(profile, binaryPath, args, actualPort)
}

// launchProcess 启动 llama-server 进程并等待就绪
func launchProcess(profile *model.DeployProfile, binaryPath string, args []string, port int) (*RunningEngine, error) {
	logPath := filepath.Join(config.LogDir(), fmt.Sprintf("llama-server-%d.log", time.Now().Unix()))
	logFile, err := os.Create(logPath)
	if err != nil {
		return nil, fmt.Errorf("failed to create log file: %w", err)
	}

	cmd := exec.Command(binaryPath, args...)
	cmd.Stdout = logFile
	cmd.Stderr = logFile
	setProcAttr(cmd)

	if err := cmd.Start(); err != nil {
		logFile.Close()
		return nil, fmt.Errorf("failed to start llama-server: %w", err)
	}

	// 监控进程退出
	exitCh := make(chan error, 1)
	go func() {
		exitCh <- cmd.Wait()
		logFile.Close()
	}()

	pidPath := filepath.Join(config.Dir(), "llama-server.pid")
	if err := os.WriteFile(pidPath, []byte(strconv.Itoa(cmd.Process.Pid)), 0644); err != nil {
		cmd.Process.Kill()
		return nil, fmt.Errorf("failed to write PID file: %w", err)
	}

	eng := &RunningEngine{
		PID:        cmd.Process.Pid,
		Port:       port,
		ModelID:    profile.ModelID,
		BinaryPath: binaryPath,
		LogPath:    logPath,
		logFile:    logFile,
	}

	fmt.Printf("Waiting for llama-server to be ready (port %d)...\n", port)

	// 等待就绪，同时监控进程是否提前退出（OOM 等）
	timeout := 90 * time.Second
	deadline := time.After(timeout)
	tick := time.NewTicker(500 * time.Millisecond)
	defer tick.Stop()

	for {
		select {
		case err := <-exitCh:
			// 进程退出了 = 启动失败（大概率 OOM）
			_ = err
			logContent := readLastLines(logPath, 20)
			return nil, fmt.Errorf("llama-server exited during startup:\n%s", logContent)
		case <-deadline:
			Stop()
			return nil, fmt.Errorf("llama-server failed to start within %s", timeout)
		case <-tick.C:
			if isPortReady("127.0.0.1", port) {
				return eng, nil
			}
		}
	}
}

// isLikelyOOM 判断启动失败是否可能是 OOM
func isLikelyOOM(err error) bool {
	msg := err.Error()
	return strings.Contains(msg, "exited during startup") ||
		strings.Contains(msg, "CUDA out of memory") ||
		strings.Contains(msg, "ggml_cuda") ||
		strings.Contains(msg, "not enough memory") ||
		strings.Contains(msg, "alloc")
}

// readLastLines 读取文件最后 n 行
func readLastLines(path string, n int) string {
	data, err := os.ReadFile(path)
	if err != nil {
		return "(无法读取日志)"
	}
	lines := strings.Split(string(data), "\n")
	if len(lines) > n {
		lines = lines[len(lines)-n:]
	}
	return strings.Join(lines, "\n")
}

// isPortReady 检查端口是否可连接
func isPortReady(host string, port int) bool {
	addr := fmt.Sprintf("%s:%d", host, port)
	conn, err := (&net.Dialer{Timeout: 500 * time.Millisecond}).Dial("tcp", addr)
	if err != nil {
		return false
	}
	conn.Close()
	return true
}

// Stop stops the running llama-server
func Stop() error {
	pidPath := filepath.Join(config.Dir(), "llama-server.pid")
	data, err := os.ReadFile(pidPath)
	if err != nil {
		if os.IsNotExist(err) {
			return fmt.Errorf("no running model found")
		}
		return fmt.Errorf("failed to read PID file: %w", err)
	}

	pid, err := strconv.Atoi(strings.TrimSpace(string(data)))
	if err != nil {
		return fmt.Errorf("invalid PID in file: %w", err)
	}

	if err := killProcess(pid); err != nil {
		return err
	}

	os.Remove(pidPath)
	return nil
}

// Status returns the status of the running engine
func Status() (*RunningEngine, error) {
	pidPath := filepath.Join(config.Dir(), "llama-server.pid")
	data, err := os.ReadFile(pidPath)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, fmt.Errorf("failed to read PID file: %w", err)
	}

	pid, err := strconv.Atoi(strings.TrimSpace(string(data)))
	if err != nil {
		return nil, fmt.Errorf("invalid PID in file: %w", err)
	}

	if !isProcessAlive(pid) {
		os.Remove(pidPath)
		return nil, nil
	}

	cfg, _ := config.Load()
	return &RunningEngine{
		PID:  pid,
		Port: cfg.LlamaPort,
	}, nil
}

// buildArgs constructs llama-server command-line arguments
// This is the fallback path when warmup cache is unavailable.
// Logic mirrors optimizer.BuildArgs to keep parameters consistent.
func buildArgs(profile *model.DeployProfile, modelPath string, port int, hw *hardware.HardwareProbe, ctxSize int) []string {
	vramMB := hw.TotalVRAM_MB() // 多卡总VRAM

	threads := threadsForMode(profile.Mode, hw)

	// KV cache 类型：基于 VRAM 计算自动选择（同 optimizer 逻辑）
	kvK, kvV := profile.SelectKVCacheType(vramMB, ctxSize)

	useMlock := shouldMlockEngine(hw, profile)

	args := []string{
		"--model", modelPath,
		"--alias", profile.ModelID,
		"--host", "127.0.0.1",
		"--port", strconv.Itoa(port),
		"--n-gpu-layers", "999",
		"--parallel", "1",
		"--cont-batching",
		"--metrics",
		"--no-webui",
		"--cache-reuse", strconv.Itoa(cacheReuseForCtx(ctxSize)),
		"-ctk", kvK,
		"-ctv", kvV,
		"--ctx-size", strconv.Itoa(ctxSize),
		"--threads", strconv.Itoa(threads),
	}

	// --kv-unified: pre-allocate KV cache as one contiguous block.
	// Blackwell (SM120) + CUDA 12.4 binary on CUDA 13.x driver causes massive over-allocation → OOM.
	if !hw.ClusterCaps().HasBlackwell {
		args = append(args, "--kv-unified")
	}

	// Flash Attention: SM75+ (Turing and newer)
	if hw.SupportsFlashAttn() {
		args = append(args, "--flash-attn", "on")
	}

	// Multi-GPU: NVLink → graph split mode; otherwise → weighted tensor-split
	if hw.GPUCount() > 1 {
		if hw.HasNVLink() {
			args = append(args, "-sm", "graph")
		} else if ts := hw.TensorSplitArg(); ts != "" {
			args = append(args, "--tensor-split", ts)
		}
	}

	// Hybrid architecture (DeltaNet/SSM): full SWA KV cache for correct long-context behavior
	if profile.IsHybrid {
		args = append(args, "--swa-full")
	}

	// mlock
	if useMlock {
		args = append(args, "--mlock")
	}

	// mmap: skip when mlock is active
	if !useMlock && shouldMmapEngine(hw, profile) {
		args = append(args, "--mmap")
	}

	// MoE offload: keep expert layers on CPU, only attention/shared layers on GPU.
	// --cpu-moe is more reliable than -ot regex.
	// --fit on lets llama.cpp handle layer allocation for both modes.
	if profile.Mode == "moe_offload" {
		args = append(args, "--cpu-moe")
		args = append(args, "--batch-size", "4096")
		args = append(args, "--ubatch-size", "512")
	} else {
		args = append(args, "--batch-size", "512")
		// 小模型（<2GB）用小 ubatch，避免 --kv-unified 预分配过多 VRAM
		ubatch := 512
		if profile.Size_GB < 2.0 {
			ubatch = 128
		}
		args = append(args, "--ubatch-size", strconv.Itoa(ubatch))
	}
	args = append(args, "--fit", "on")

	return args
}

// threadsForMode returns the optimal thread count based on inference mode.
func threadsForMode(mode string, hw *hardware.HardwareProbe) int {
	if mode == "moe_offload" {
		t := hw.CPU.Cores / 2
		if t < 4 {
			t = 4
		}
		return t
	}
	return 2
}

// cacheReuseForCtx returns dynamic cache-reuse size based on context.
func cacheReuseForCtx(ctxSize int) int {
	switch {
	case ctxSize >= 32768:
		return 1024
	case ctxSize >= 8192:
		return 512
	default:
		return 256
	}
}

// shouldMlockEngine mirrors optimizer.shouldMlock for the fallback path.
func shouldMlockEngine(hw *hardware.HardwareProbe, profile *model.DeployProfile) bool {
	totalMB := float64(hw.RAM.Total_MB)
	freeMB := float64(hw.RAM.Free_MB)
	if totalMB == 0 {
		return false
	}
	var modelRAM_MB float64
	if profile.Mode == "moe_offload" {
		modelRAM_MB = profile.Size_GB * 1024 * 0.9
	} else {
		modelRAM_MB = profile.Size_GB * 1024 * 0.1
	}
	return (freeMB - modelRAM_MB) > totalMB*0.3
}

// shouldMmapEngine mirrors optimizer.shouldMmap for the fallback path.
func shouldMmapEngine(hw *hardware.HardwareProbe, profile *model.DeployProfile) bool {
	freeMB := float64(hw.RAM.Free_MB)
	modelMB := profile.Size_GB * 1024
	return modelMB < freeMB*0.7
}

// buildOOMError 构建详细的 OOM 错误信息，建议根据实际情况动态生成
func buildOOMError(profile *model.DeployProfile, hw *hardware.HardwareProbe, attempts int) error {
	vramMB := hw.TotalVRAM_MB()
	modelMB := int(profile.Size_GB * 1024)
	kvMB := profile.EstimateKVCacheMB(4096, "q4_0") * 2 // 最小 KV cache
	needMB := modelMB + kvMB + 1024                      // 模型 + KV + overhead

	gpu := hw.PrimaryGPU()
	gpuName := "GPU"
	if gpu != nil {
		gpuName = gpu.Name
	}

	msg := fmt.Sprintf("连续 %d 次启动失败，即使最小上下文(4K)也无法运行\n\n", attempts)
	msg += fmt.Sprintf("  %s: %d MB VRAM\n", gpuName, vramMB)
	msg += fmt.Sprintf("  模型 %s: ~%d MB\n", profile.DisplayName, modelMB)
	msg += fmt.Sprintf("  KV cache (4K, q4_0): ~%d MB\n", kvMB)
	msg += fmt.Sprintf("  预估总需: ~%d MB\n\n", needMB)

	if needMB > vramMB {
		msg += fmt.Sprintf("  差额: %d MB\n\n", needMB-vramMB)
	}

	msg += "  建议:\n"
	vramGB := float64(vramMB) / 1024.0

	if profile.Size_GB > vramGB*0.7 {
		// 模型本身偏大
		msg += "  1. 选择更小的量化 (Q4_K_M 或 Q2_K)\n"
		if profile.Arch != "moe" && profile.Size_GB > 10 {
			// dense 大模型，建议换 MoE 架构
			msg += "  2. 换用 MoE 架构模型（如 Qwen3-30B-A3B），expert 层自动放 CPU RAM\n"
		} else {
			msg += "  2. 选择更小的模型\n"
		}
	} else {
		// 模型小但还是 OOM → 参数配置问题
		msg += "  1. 运行 kaiwu run " + profile.ModelID + " --reset 重新探测参数\n"
		msg += "  2. 模型较小但仍 OOM，可能是参数配置问题，请升级到最新版本\n"
	}

	return fmt.Errorf("%s", msg)
}
