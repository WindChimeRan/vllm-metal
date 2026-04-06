# macOS Metal Memory Pressure Tuning for vllm-metal

## Background

On Apple Silicon, **GPU memory IS wired memory** — there is no separate VRAM pool.
The OS has multiple layers of memory reclamation that can cause sudden memory drops
during large-scale inference. Setting `iogpu.disable_wired_collector=1` alone is
insufficient because it only disables one of several reclamation mechanisms.

## Sysctl Settings (run before starting the server)

All settings are volatile and reset on reboot.

```bash
# 1. Raise the hard ceiling (e.g., 56GB out of 64GB, leaving 8GB for OS)
sudo sysctl iogpu.wired_limit_mb=57344

# 2. Disable the wired collector
sudo sysctl iogpu.disable_wired_collector=1

# 3. Disable the dynamic low water mark adjustment
sudo sysctl iogpu.dynamic_lwm=0
```

### Verify settings took effect

```bash
sysctl iogpu.wired_limit_mb iogpu.disable_wired_collector iogpu.dynamic_lwm
```

Expected output:
```
iogpu.wired_limit_mb: 57344
iogpu.disable_wired_collector: 1
iogpu.dynamic_lwm: 0
```

## What Each Setting Does

| Setting | Default | Purpose |
|---|---|---|
| `iogpu.wired_limit_mb` | 0 (= ~75% of RAM) | Hard ceiling on how much memory can be wired for GPU use |
| `iogpu.disable_wired_collector` | 0 (enabled) | When 1, disables the periodic background reclamation of GPU wired memory |
| `iogpu.dynamic_lwm` | 1 (enabled) | When 1, system dynamically adjusts the low water mark target; set to 0 to prevent shifting targets |

## Important: vllm-metal's Own Budget Cap

Even with all sysctl flags set, vllm-metal's KV cache budget is capped by
`max_recommended_working_set_size` (~75% of RAM). This is read in
`vllm_metal/v1/worker.py:208`:

```python
metal_limit = int(device_info.get("max_recommended_working_set_size", 0))
```

The KV budget formula (`worker.py:170-176`):

```
KV_budget = (metal_limit * fraction) - model_memory - overhead
```

To push allocation beyond the default 75% ceiling, set `VLLM_METAL_MEMORY_FRACTION`
higher (the default for paged attention is 0.9, applied to the ~75% metal_limit):

```bash
# Example: use 95% of the reported metal_limit for KV cache + model
export VLLM_METAL_MEMORY_FRACTION=0.95
```

Note: this fraction is applied to `max_recommended_working_set_size`, NOT total RAM.
On a 64GB machine: `48GB * 0.95 = ~45.6GB` usable for model + KV cache + overhead.

## Recommended Benchmark Invocation

```bash
# Set sysctl first (see above), then:
export VLLM_METAL_MEMORY_FRACTION=0.95
export VLLM_METAL_USE_PAGED_ATTENTION=1

# Start the server as usual
python -m vllm_metal.entrypoints.openai.api_server ...
```

## Monitoring During Benchmark

Watch wired memory in a separate terminal:

```bash
# Live memory stats (updates every 2 seconds)
while true; do
    echo "$(date '+%H:%M:%S') | $(memory_pressure -Q | head -1) | wired: $(vm_stat | awk '/wired/ {print $4}' | tr -d '.')"
    sleep 2
done
```

Or use Activity Monitor > Memory tab — watch the "Wired" value.

## Persist Across Reboots (Optional)

Create a LaunchDaemon to apply settings at boot:

```bash
sudo tee /Library/LaunchDaemons/com.vllm.memory-tuning.plist > /dev/null << 'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.vllm.memory-tuning</string>
    <key>ProgramArguments</key>
    <array>
        <string>/bin/bash</string>
        <string>-c</string>
        <string>sysctl iogpu.wired_limit_mb=57344 iogpu.disable_wired_collector=1 iogpu.dynamic_lwm=0</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
</dict>
</plist>
EOF

sudo launchctl load /Library/LaunchDaemons/com.vllm.memory-tuning.plist
```
