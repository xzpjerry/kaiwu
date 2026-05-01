package optimizer

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/val1813/kaiwu/internal/hardware"
	"github.com/val1813/kaiwu/internal/model"
)

func TestBuildArgsSkipsUnsupportedSpeculativeFlags(t *testing.T) {
	binaryPath := fakeLlamaServer(t, "usage: llama-server --lookup-cache-static FNAME\n")
	hw := testHardware()

	withMTP := &model.DeployProfile{ModelID: "qwen-next", NativeMTP: true}
	args := BuildArgs(withMTP, binaryPath, "/tmp/model.gguf", 12345, hw, 4096, 512, 128)
	if containsArg(args, "--num-speculative-tokens") {
		t.Fatalf("BuildArgs included unsupported --num-speculative-tokens: %v", args)
	}

	withoutMTP := &model.DeployProfile{ModelID: "dense"}
	args = BuildArgs(withoutMTP, binaryPath, "/tmp/model.gguf", 12345, hw, 4096, 512, 128)
	if containsArg(args, "--lookup") {
		t.Fatalf("BuildArgs included unsupported --lookup: %v", args)
	}
}

func fakeLlamaServer(t *testing.T, help string) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "llama-server")
	script := "#!/bin/sh\nif [ \"$1\" = \"--help\" ]; then\n  cat <<'EOF'\n" + help + "EOF\n  exit 0\nfi\nexit 0\n"
	if err := os.WriteFile(path, []byte(script), 0755); err != nil {
		t.Fatal(err)
	}
	return path
}

func testHardware() *hardware.HardwareProbe {
	return &hardware.HardwareProbe{
		GPUs: []hardware.GPUInfo{{Name: "AMD GPU", VRAM_MB: 8176}},
		CPU:  hardware.CPUInfo{Cores: 24},
		RAM:  hardware.RAMInfo{Total_MB: 23998, Free_MB: 22511},
	}
}

func containsArg(args []string, want string) bool {
	for _, arg := range args {
		if arg == want {
			return true
		}
	}
	return false
}
