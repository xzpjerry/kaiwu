package optimizer

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/val1813/kaiwu/internal/config"
	"github.com/val1813/kaiwu/internal/engine"
	"github.com/val1813/kaiwu/internal/hardware"
	"github.com/val1813/kaiwu/internal/model"
)

// minAcceptableTPS is the minimum decode speed for acceptable user experience.
// Only applies to full_gpu mode — MoE offload uses threshold=0 (find largest ctx regardless of speed).
const minAcceptableTPS = 18.0

// maxUpwardProbes limits how many times we double ctx after the first success.
// Prevents 8K→16K→32K→64K→128K→256K chains on large VRAM (each costs 30-60s).
const maxUpwardProbes = 3

// OptimizedProfile is the result of warmup benchmark
type OptimizedProfile struct {
	ModelID     string   `json:"model_id"`
	HardwareFP  string   `json:"hardware_fp"`
	Quant       string   `json:"quant"`
	Mode        string   `json:"mode"`
	MeasuredTPS float64  `json:"measured_tps"`
	VRAMUsed_MB int      `json:"vram_used_mb"`
	LaunchArgs  []string `json:"launch_args"`
	CreatedAt   string   `json:"created_at"`
}

// ClearProfileCache deletes the cached profile for a given model+hardware combo.
func ClearProfileCache(modelID string, hw *hardware.HardwareProbe) error {
	fingerprint := hw.Fingerprint()
	profilePath := filepath.Join(config.ProfileDir(), fmt.Sprintf("%s_%s.json", modelID, fingerprint))
	if err := os.Remove(profilePath); err != nil {
		if os.IsNotExist(err) {
			return nil // nothing to clear
		}
		return err
	}
	return nil
}

// Warmup runs the warmup benchmark and returns optimized parameters.
//
// ┌─────────────────────────────────────────────────────────────────┐
// │ Warmup Flow                                                     │
// │                                                                 │
// │ 1. Cache check                                                  │
// │    - CtxOverride > 0 → skip cache                               │
// │    - Valid cache → return immediately (2s startup)               │
// │    - --fast + no cache → error                                  │
// │                                                                 │
// │ 2. Phase 1: Coarse search (power-of-2 steps, max 8 probes)     │
// │    Start from IdealStartCtx×2 (full_gpu) or nativeMax (MoE)    │
// │    a) OOM       → record failedCtx, halve, retry                │
// │    b) Too slow  → record failedCtx, halve, retry                │
// │    c) Success   → record bestCtx, double up (max 3 times)       │
// │    d) Double fails → stop, we have [bestCtx, failedCtx] range   │
// │                                                                 │
// │ 3. Phase 1.5: Fine search (4K precision binary search)          │
// │    Between bestCtx and failedCtx, binary search for real max    │
// │    Only runs when failedCtx > bestCtx (gap to explore)          │
// │                                                                 │
// │ 4. Phase 2: ubatch tuning                                       │
// │    Test 128 vs 512, pick faster. Uses bestCtx from above.       │
// │                                                                 │
// │ 5. Save profile to cache                                        │
// └─────────────────────────────────────────────────────────────────┘
func Warmup(profile *model.DeployProfile, binaryPath, modelPath string, hw *hardware.HardwareProbe, fast bool) (*OptimizedProfile, error) {
	fingerprint := hw.Fingerprint()
	profilePath := filepath.Join(config.ProfileDir(), fmt.Sprintf("%s_%s.json", profile.ModelID, fingerprint))

	// User specified --ctx-size → skip cache, use their value directly
	if profile.CtxOverride > 0 {
		fmt.Printf("      用户指定 ctx=%d，跳过缓存\n", profile.CtxOverride)
	} else {
		// Check cache (spec: second launch should be 2s)
		if cached, err := loadCachedProfile(profilePath); err == nil {
			if isCacheValid(cached, fingerprint) {
				created, _ := time.Parse(time.RFC3339, cached.CreatedAt)
				age := time.Since(created)
				ageStr := fmt.Sprintf("%.0f 天前", age.Hours()/24)
				if age.Hours() < 24 {
					ageStr = fmt.Sprintf("%.0f 小时前", age.Hours())
				}
				// Extract ctx from cached args for display
				cachedCtx := extractArgValue(cached.LaunchArgs, "--ctx-size")
				fmt.Printf("      使用上次配置（%s ctx · %.1f tok/s · %s）\n", cachedCtx, cached.MeasuredTPS, ageStr)
				return cached, nil
			}
			fmt.Printf("      Cache expired, re-running warmup\n")
		}
	}

	// --fast with no cache: skip warmup entirely
	if fast {
		fmt.Printf("      No cached profile, using defaults\n")
		return nil, fmt.Errorf("no cached profile available")
	}

	cfg, _ := config.Load()
	port := cfg.LlamaPort + 10

	// RAM safety check
	if warning := checkRAMSafety(hw, profile); warning != "" {
		fmt.Printf("      %s\n", warning)
	}

	// --- Phase 1: probe max ctx (speed-aware) ---
	nativeMax := profile.NativeCtx
	if nativeMax <= 0 {
		nativeMax = 131072
	}

	batchSize, ubatchSize := 512, 128
	if profile.Mode == "moe_offload" {
		batchSize, ubatchSize = 4096, 512
	}

	// full_gpu: speed correlates with ctx size, threshold finds the speed/ctx balance point.
	// moe_offload: speed is PCIe-bandwidth-limited, not ctx-limited. Dropping ctx from 128K
	// to 4K only improves speed by ~20-30%, never enough to cross a 18 tok/s threshold.
	// Disable the threshold for MoE — just find the largest ctx that fits in VRAM.
	tpsThreshold := minAcceptableTPS
	if profile.Mode == "moe_offload" {
		tpsThreshold = 0
	}

	var bestCtx int
	var bestTPS float64
	var bestArgs []string
	var bestVRAM int

	if profile.CtxOverride > 0 {
		// User override: use exact value, no alignment
		ctxFixed := profile.CtxOverride
		// Clamp to model's native max
		if profile.NativeCtx > 0 && ctxFixed > profile.NativeCtx {
			fmt.Printf("      Warning: ctx %d 超过模型最大值 %d，已截断\n", ctxFixed, profile.NativeCtx)
			ctxFixed = profile.NativeCtx
		}
		fmt.Printf("      User override: ctx=%s ... ", fmtCtx(ctxFixed))
		args := BuildArgs(profile, modelPath, port, hw, ctxFixed, batchSize, ubatchSize)
		tps, vram, err := runBenchmarkRound(binaryPath, args, port)
		if err != nil {
			return nil, fmt.Errorf("user-specified ctx=%s failed to start (OOM?)", fmtCtx(ctxFixed))
		}
		fmt.Printf("%.1f tok/s\n", tps)
		bestCtx = ctxFixed
		bestTPS = tps
		bestArgs = args
		bestVRAM = vram
	} else {
		// --- Phase 1: coarse search (power-of-2 steps, max 8 probes) ---
		// Starting point:
		//   full_gpu:     IdealStartCtx×2 (oobabooga formula + 1 step headroom)
		//   moe_offload:  nativeMax (formula wrong for MoE, GPU only has KV cache)
		var startCtx int
		if profile.Mode == "moe_offload" {
			startCtx = nativeMax
		} else {
			ideal := engine.IdealStartCtx(profile, hw)
			startCtx = ideal * 2
			if startCtx > nativeMax {
				startCtx = nativeMax
			}
		}
		startCtx = alignToPow2(startCtx)

		var failedCtx int // smallest ctx that failed (OOM or too slow)
		upwardProbes := 0 // count of upward doublings after first success
		ctxTry := startCtx
		for attempt := 1; attempt <= 8; attempt++ {
			fmt.Printf("      Probe %d: ctx=%s ... ", attempt, fmtCtx(ctxTry))
			args := BuildArgs(profile, modelPath, port, hw, ctxTry, batchSize, ubatchSize)
			tps, vram, err := runBenchmarkRound(binaryPath, args, port)

			if err != nil {
				// OOM: record ceiling, halve and retry downward
				fmt.Printf("OOM\n")
				if failedCtx == 0 || ctxTry < failedCtx {
					failedCtx = ctxTry
				}
				ctxTry /= 2
				if ctxTry < 4096 {
					ctxTry = 4096
				}
				if ctxTry == 4096 && attempt > 1 {
					break
				}
				continue
			}

			if tps < tpsThreshold {
				// Too slow: record ceiling, keep as fallback, halve
				fmt.Printf("%.1f tok/s (< %.0f, too slow)\n", tps, tpsThreshold)
				if failedCtx == 0 || ctxTry < failedCtx {
					failedCtx = ctxTry
				}
				if bestCtx == 0 || tps > bestTPS {
					bestCtx = ctxTry
					bestTPS = tps
					bestArgs = args
					bestVRAM = vram
				}
				ctxTry /= 2
				if ctxTry < 4096 {
					ctxTry = 4096
				}
				if ctxTry == 4096 && attempt > 1 {
					break
				}
				continue
			}

			// Success: speed >= threshold
			fmt.Printf("%.1f tok/s\n", tps)
			bestCtx = ctxTry
			bestTPS = tps
			bestArgs = args
			bestVRAM = vram

			// Try doubling to find the ceiling (max 3 upward probes)
			if failedCtx == 0 && ctxTry < nativeMax && upwardProbes < maxUpwardProbes {
				upwardProbes++
				nextCtx := alignToPow2(ctxTry * 2)
				if nextCtx > nativeMax {
					nextCtx = alignToPow2(nativeMax)
				}
				if nextCtx > ctxTry {
					ctxTry = nextCtx
					continue
				}
			}
			break
		}

		// --- Phase 1.5: fine search (4K precision binary search) ---
		// Between bestCtx (works) and failedCtx (too slow/OOM), find the real max.
		if bestCtx > 0 && failedCtx > bestCtx {
			lo := bestCtx
			hi := failedCtx
			for hi-lo > 4096 {
				mid := ((lo + hi) / 2 / 1024) * 1024 // align to 1K boundary
				if mid <= lo || mid >= hi {
					break
				}
				fmt.Printf("      Fine: ctx=%s ... ", fmtCtx(mid))
				args := BuildArgs(profile, modelPath, port, hw, mid, batchSize, ubatchSize)
				tps, vram, err := runBenchmarkRound(binaryPath, args, port)
				if err != nil {
					fmt.Printf("OOM\n")
					hi = mid
					continue
				}
				if tps < tpsThreshold {
					fmt.Printf("%.1f tok/s (too slow)\n", tps)
					hi = mid
					continue
				}
				fmt.Printf("%.1f tok/s\n", tps)
				lo = mid
				bestCtx = mid
				bestTPS = tps
				bestArgs = args
				bestVRAM = vram
			}
		}
	}

	if bestCtx == 0 {
		return nil, fmt.Errorf("all ctx probes failed (tried down to 4K)")
	}

	// --- Phase 2: ubatch 探测 ---
	// 低带宽卡（< 200 GB/s）只测 128，避免大 ubatch 加剧带宽瓶颈
	// 高带宽卡测 128 和 512，选速度快的
	ubatchCandidates := []int{128, 512}
	if bw := hw.PrimaryGPU(); bw != nil && bw.MemBandwidth_GBs > 0 && bw.MemBandwidth_GBs < 200 {
		ubatchCandidates = []int{128}
	}
	var ubBestTPS float64
	var ubBestArgs []string
	var ubBestVRAM int
	fmt.Printf("      Tune ubatch: ")
	for _, ub := range ubatchCandidates {
		args2 := BuildArgs(profile, modelPath, port, hw, bestCtx, 512, ub)
		tps2, vram2, err := runBenchmarkRound(binaryPath, args2, port)
		if err != nil {
			fmt.Printf("ub=%d failed; ", ub)
			continue
		}
		fmt.Printf("ub=%d → %.1f tok/s; ", ub, tps2)
		if tps2 > ubBestTPS {
			ubBestTPS = tps2
			ubBestArgs = args2
			ubBestVRAM = vram2
		}
	}
	fmt.Println()
	// Phase 2 结果覆盖 Phase 1（只要有成功的测试）
	if ubBestTPS > 0 {
		bestTPS = ubBestTPS
		bestArgs = ubBestArgs
		bestVRAM = ubBestVRAM
	}

	fmt.Printf("      ✓ %.1f tok/s @ %s ctx\n", bestTPS, fmtCtx(bestCtx))
	if profile.Mode == "moe_offload" {
		fmt.Printf("      ℹ MoE offload · speed limited by PCIe bandwidth, not context size\n")
	}

	// Save profile
	optimized := &OptimizedProfile{
		ModelID:     profile.ModelID,
		HardwareFP:  fingerprint,
		Quant:       profile.Quant,
		Mode:        profile.Mode,
		MeasuredTPS: bestTPS,
		VRAMUsed_MB: bestVRAM,
		LaunchArgs:  bestArgs,
		CreatedAt:   time.Now().Format(time.RFC3339),
	}

	if err := saveProfile(optimized, profilePath); err != nil {
		fmt.Printf("      Warning: failed to save profile: %v\n", err)
	} else {
		fmt.Printf("      Saved profile: %s\n", profilePath)
	}

	return optimized, nil
}

// alignToPow2 rounds down to the nearest power of 2 (min 4096).
func alignToPow2(n int) int {
	powers := []int{524288, 262144, 131072, 65536, 32768, 16384, 8192, 4096}
	for _, p := range powers {
		if n >= p {
			return p
		}
	}
	return 4096
}

// BuildArgs constructs llama-server arguments with explicit ctx size and batch sizes.
func BuildArgs(profile *model.DeployProfile, modelPath string, port int, hw *hardware.HardwareProbe, ctxSize, batchSize, ubatchSize int) []string {
	vramMB := hw.TotalVRAM_MB() // 多卡总VRAM

	// KV cache 类型：基于 VRAM 计算自动选择最快的类型
	// 优先 f16（最快），装不下就降到 q8_0+q4_0，再不行用 iso3
	kvK, kvV := profile.SelectKVCacheType(vramMB, ctxSize)

	// Parallel slots: 单用户场景用 1 slot（速度优先）
	// 8GB 以下必须 1 slot，大显存可以考虑多 slot 但目前统一用 1
	parallel := "1"

	useMlock := shouldMlock(hw, profile)

	args := []string{
		"--model", modelPath,
		"--host", "127.0.0.1",
		"--port", strconv.Itoa(port),
		"--n-gpu-layers", "999",
		"--parallel", parallel,
		"--cont-batching",
		"--metrics",
		"--no-webui",
		"--cache-reuse", strconv.Itoa(cacheReuseForCtx(ctxSize)),
		"-ctk", kvK,
		"-ctv", kvV,
		"--ctx-size", strconv.Itoa(ctxSize),
		"--threads", strconv.Itoa(threadsForMode(profile.Mode, hw)),
		"--batch-size", strconv.Itoa(batchSize),
		"--ubatch-size", strconv.Itoa(ubatchSize),
		"--kv-unified",
	}

	// MoE offload: keep expert layers on CPU, only attention/shared layers on GPU.
	// --cpu-moe is more reliable than -ot regex.
	// Keep --fit on so llama.cpp does the final precise layer allocation.
	if profile.Mode == "moe_offload" {
		args = append(args, "--cpu-moe")
	}
	args = append(args, "--fit", "on")

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

	// mlock: lock model in RAM when headroom is sufficient
	if useMlock {
		args = append(args, "--mlock")
	}

	// mmap: accelerate model loading via memory-mapped I/O
	// Skip when mlock is active (mlock+mmap can conflict on some systems)
	if !useMlock && shouldMmap(hw, profile) {
		args = append(args, "--mmap")
	}

	return args
}

// runBenchmarkRound starts llama-server, runs a warmup prompt, measures tok/s
func runBenchmarkRound(binaryPath string, args []string, port int) (tps float64, vramMB int, err error) {
	// Start llama-server
	eng, err := startBenchServer(binaryPath, args, port)
	if err != nil {
		return 0, 0, err
	}
	defer stopBenchServer(eng)

	// Send warmup prompt and measure
	tps, err = measureTPS(port)
	if err != nil {
		return 0, 0, err
	}

	// Get VRAM usage from metrics
	vramMB = getVRAMFromMetrics(port)

	return tps, vramMB, nil
}

// startBenchServer starts llama-server for benchmarking
func startBenchServer(binaryPath string, args []string, port int) (*os.Process, error) {
	// Create log file for debugging
	logPath := filepath.Join(config.LogDir(), fmt.Sprintf("warmup-%d.log", time.Now().Unix()))
	logFile, err := os.Create(logPath)
	if err != nil {
		logFile = nil
	}

	cmd := exec.Command(binaryPath, args...)
	if logFile != nil {
		cmd.Stdout = logFile
		cmd.Stderr = logFile
	}
	engine.SetProcAttr(cmd)

	if err := cmd.Start(); err != nil {
		if logFile != nil {
			logFile.Close()
		}
		return nil, fmt.Errorf("failed to start benchmark server: %w", err)
	}

	// Close log file when process exits
	go func() {
		cmd.Wait()
		if logFile != nil {
			logFile.Close()
		}
	}()

	// Wait for health endpoint
	deadline := time.Now().Add(180 * time.Second) // MoE models need longer to load (~13GB from RAM)
	for time.Now().Before(deadline) {
		resp, err := http.Get(fmt.Sprintf("http://127.0.0.1:%d/health", port))
		if err == nil {
			resp.Body.Close()
			if resp.StatusCode == http.StatusOK {
				return cmd.Process, nil
			}
		}
		time.Sleep(1 * time.Second)
	}

	cmd.Process.Kill()
	return nil, fmt.Errorf("benchmark server failed to start within 60s (log: %s)", logPath)
}

// stopBenchServer stops the benchmark server
func stopBenchServer(proc *os.Process) {
	if proc != nil {
		proc.Kill()
		proc.Wait()
	}
}

// measureTPS sends a prompt and measures decode tokens per second
func measureTPS(port int) (float64, error) {
	// Use pure ASCII prompt to avoid UTF-8 encoding issues on Windows bash
	reqBody := `{
		"model": "test",
		"messages": [{"role": "user", "content": "Write a Python quicksort implementation with detailed comments and edge case handling."}],
		"max_tokens": 200,
		"stream": false
	}`

	start := time.Now()
	resp, err := http.Post(
		fmt.Sprintf("http://127.0.0.1:%d/v1/chat/completions", port),
		"application/json",
		strings.NewReader(reqBody),
	)
	if err != nil {
		return 0, fmt.Errorf("benchmark request failed: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return 0, fmt.Errorf("failed to read response: %w", err)
	}
	elapsed := time.Since(start)

	// Parse response to get token count
	var result struct {
		Usage struct {
			CompletionTokens int `json:"completion_tokens"`
		} `json:"usage"`
	}
	if err := json.Unmarshal(body, &result); err != nil {
		return 0, fmt.Errorf("failed to parse response: %w", err)
	}

	tokens := result.Usage.CompletionTokens
	if tokens == 0 {
		return 0, fmt.Errorf("no tokens generated")
	}

	tps := float64(tokens) / elapsed.Seconds()
	return tps, nil
}

// getVRAMFromMetrics reads VRAM usage from llama-server metrics endpoint
func getVRAMFromMetrics(port int) int {
	resp, err := http.Get(fmt.Sprintf("http://127.0.0.1:%d/metrics", port))
	if err != nil {
		return 0
	}
	defer resp.Body.Close()

	body, _ := io.ReadAll(resp.Body)
	// Look for VRAM metric in Prometheus format
	for _, line := range strings.Split(string(body), "\n") {
		if strings.HasPrefix(line, "llama_vram_usage_bytes") {
			parts := strings.Fields(line)
			if len(parts) >= 2 {
				val, _ := strconv.ParseFloat(parts[1], 64)
				return int(val / (1024 * 1024))
			}
		}
	}
	return 0
}

func loadCachedProfile(path string) (*OptimizedProfile, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var profile OptimizedProfile
	if err := json.Unmarshal(data, &profile); err != nil {
		return nil, err
	}
	return &profile, nil
}

// isCacheValid checks if cached profile is still valid (fingerprint match + < 30 days old)
func isCacheValid(cached *OptimizedProfile, currentFingerprint string) bool {
	// Check fingerprint match
	if cached.HardwareFP != currentFingerprint {
		return false
	}
	// Check age (30 days)
	created, err := time.Parse(time.RFC3339, cached.CreatedAt)
	if err != nil {
		return false
	}
	return time.Since(created) < 30*24*time.Hour
}

func saveProfile(profile *OptimizedProfile, path string) error {
	data, err := json.MarshalIndent(profile, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, data, 0644)
}

// threadsForMode returns the optimal thread count based on inference mode.
// full_gpu: GPU does all the work, CPU just orchestrates — 2 threads suffice.
// moe_offload: CPU handles expert layers, needs real parallelism.
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

// checkRAMSafety warns when free RAM is too low for comfortable operation.
// Reserve = max(Total_MB * 20%, 2048 MB) for system stability.
func checkRAMSafety(hw *hardware.HardwareProbe, profile *model.DeployProfile) string {
	totalMB := hw.RAM.Total_MB
	freeMB := hw.RAM.Free_MB

	reserveMB := totalMB / 5
	if reserveMB < 2048 {
		reserveMB = 2048
	}

	var modelRAM_MB uint64
	if profile.Mode == "moe_offload" {
		modelRAM_MB = uint64(profile.Size_GB * 1024 * 0.9)
	} else {
		modelRAM_MB = uint64(profile.Size_GB * 1024 * 0.1)
	}

	needed := modelRAM_MB + reserveMB
	if freeMB < needed {
		return fmt.Sprintf("Warning: RAM tight: %.1f GB free, need ~%.1f GB (%.1f GB model + %.1f GB reserve)",
			float64(freeMB)/1024,
			float64(needed)/1024,
			float64(modelRAM_MB)/1024,
			float64(reserveMB)/1024)
	}
	return ""
}

// shouldMlock returns true when there's enough RAM headroom to safely lock
// the model in memory. mlock prevents swapping model pages to disk.
// Only enabled when remaining RAM after model load > 30% of total.
func shouldMlock(hw *hardware.HardwareProbe, profile *model.DeployProfile) bool {
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

	remainingAfterLoad := freeMB - modelRAM_MB
	return remainingAfterLoad > totalMB*0.3
}

// cacheReuseForCtx returns dynamic cache-reuse size based on context.
// Larger ctx → longer system prompts → more tokens worth caching.
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

// shouldMmap returns true when memory-mapped I/O should be used for faster model loading.
// Enabled when model size < 70% of available RAM (avoids swap pressure).
func shouldMmap(hw *hardware.HardwareProbe, profile *model.DeployProfile) bool {
	freeMB := float64(hw.RAM.Free_MB)
	modelMB := profile.Size_GB * 1024
	return modelMB < freeMB*0.7
}

// fmtCtx formats a ctx size for human display: "8K", "12K", "65536" etc.
func fmtCtx(ctx int) string {
	if ctx >= 1024 && ctx%1024 == 0 {
		return fmt.Sprintf("%dK", ctx/1024)
	}
	return strconv.Itoa(ctx)
}

// extractArgValue extracts the value of a flag from args (e.g., "--ctx-size" → "8K")
func extractArgValue(args []string, flag string) string {
	for i, a := range args {
		if a == flag && i+1 < len(args) {
			val := args[i+1]
			if flag == "--ctx-size" {
				if n, err := strconv.Atoi(val); err == nil {
					return fmtCtx(n)
				}
			}
			return val
		}
	}
	return ""
}
