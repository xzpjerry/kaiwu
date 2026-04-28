package optimizer

import (
	"github.com/val1813/kaiwu/internal/engine"
	"github.com/val1813/kaiwu/internal/hardware"
	"github.com/val1813/kaiwu/internal/model"
)

// StartingParams holds initial parameter recommendations
type StartingParams struct {
	Mode       string
	OTFlags    string
	BatchSizes []int
	UBatchSize int
	Threads    int
	CtxSize    int
	KVCacheK   string
	KVCacheV   string
}

// DeriveStartingParams applies the three parameter rules from spec
func DeriveStartingParams(hw *hardware.HardwareProbe, profile *model.DeployProfile) StartingParams {
	vramMB := hw.TotalVRAM_MB() // 多卡总VRAM

	isMoE := profile.Arch == "moe"

	ctxSize := engine.IdealStartCtx(profile, hw)

	// KV cache 类型：基于 VRAM 计算自动选择（同 warmup/runner）
	kvK, kvV := profile.SelectKVCacheType(vramMB, ctxSize)

	// MoE 模型始终走 moe_offload：
	// 即使 VRAM 够装整个模型权重，full_gpu 模式下 128 个 expert 层的 compute buffer
	// 会占用巨量显存（远超模型权重本身），导致 OOM。
	// expert 层放 CPU 不影响速度（只有少数 active expert 被激活，瓶颈是 PCIe 带宽），
	// 还能省出大量显存给 KV cache 跑更大 ctx。
	if isMoE {
		moeThreads := hw.CPU.Cores / 2
		if moeThreads < 4 {
			moeThreads = 4
		}
		return StartingParams{
			Mode:       "moe_offload",
			OTFlags:    profile.OTFlags,
			BatchSizes: []int{1024, 2048, 4096},
			UBatchSize: 512,
			Threads:    moeThreads,
			CtxSize:    ctxSize,
			KVCacheK:   kvK,
			KVCacheV:   kvV,
		}
	}

	return StartingParams{
		Mode:       "full_gpu",
		BatchSizes: []int{256, 512, 1024},
		UBatchSize: 512, // 默认 512，warmup 会实测 128 vs 512
		Threads:    2,
		CtxSize:    ctxSize,
		KVCacheK:   kvK,
		KVCacheV:   kvV,
	}
}
