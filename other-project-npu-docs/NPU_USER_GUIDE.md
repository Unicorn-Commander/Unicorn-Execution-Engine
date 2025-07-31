# NPU User Guide for Meeting-Ops
*A comprehensive guide for using NPU-accelerated transcription*

## Table of Contents
1. [Overview](#overview)
2. [System Requirements](#system-requirements)
3. [Quick Start](#quick-start)
4. [Frontend Integration](#frontend-integration)
5. [Performance Monitoring](#performance-monitoring)
6. [Troubleshooting](#troubleshooting)
7. [FAQ](#faq)

## Overview

The Meeting-Ops system uses AMD Phoenix NPU hardware acceleration to deliver real-time transcription with exceptional performance. The NPU provides:

- **2,985x real-time performance** - Process 8.7 minutes of audio in 0.175 seconds
- **Hardware-only operation** - No CPU fallback or emulation
- **Direct hardware access** - Custom runtime bypasses vendor tools
- **Mandatory acceleration** - System requires NPU for operation

## System Requirements

### Hardware
- AMD Ryzen 7040/8040 series processor with NPU
- Minimum 16GB RAM
- USB or line-in audio input device

### Software
- Linux kernel 6.14+ (amdxdna driver)
- User must be in 'render' group
- Meeting-Ops backend running on port 9050

### Verification Commands
```bash
# Check NPU device
ls -la /dev/accel/accel0

# Verify user permissions
groups | grep render

# Check kernel module
lsmod | grep amdxdna
```

## Quick Start

### 1. Start the Backend
```bash
cd backend
python -m uvicorn main:app --host 0.0.0.0 --port 9050
```

Look for these confirmation messages:
```
✅ NPU device accessible at /dev/accel/accel0
✅ NPU AIE Version: 1.1
✅ NPU Accelerator ready - HARDWARE MODE ONLY
```

### 2. Access the Frontend
```bash
# Navigate to frontend
cd frontend

# Ensure correct backend URL
echo "VITE_API_URL=http://localhost:9050" > .env

# Start frontend
npm run dev
```

### 3. Verify NPU Status
Open the dashboard and check the System Monitor widget:
- NPU Status: ✅ Active
- NPU Device: AMD Phoenix NPU
- AIE Version: 1.1
- Acceleration: 16 TOPS INT8

## Frontend Integration

### Dashboard Components

All frontend components now connect to the NPU-accelerated backend:

#### 1. Recording Controls
```jsx
// The recording button now uses NPU acceleration
<button onClick={startRecording}>
  Start Recording (NPU Accelerated)
</button>
```

#### 2. System Monitor
The System Monitor displays real-time NPU status:
```jsx
<SystemMonitor />
// Shows:
// - NPU Device: AMD Phoenix NPU ✅
// - Hardware Mode: Active
// - Processing Speed: 2,985x real-time
```

#### 3. Session List
Sessions show NPU acceleration status:
```jsx
<SessionList />
// Each session displays:
// - "NPU Accelerated" badge
// - Processing time metrics
// - Real-time factor
```

#### 4. Transcription Viewer
Live transcriptions show NPU performance:
```jsx
<TranscriptionViewer sessionId={sessionId} />
// Displays:
// - Real-time transcription text
// - NPU processing indicators
// - Latency metrics
```

### API Endpoints

All API calls automatically use NPU acceleration:

```javascript
// Start recording with NPU
const response = await fetch('/api/recording-sessions', {
  method: 'POST',
  headers: { 'Authorization': `Bearer ${token}` }
});

// The backend will:
// 1. Initialize NPU hardware
// 2. Create DMA buffers
// 3. Load Whisper model to NPU
// 4. Process audio in real-time
```

### WebSocket Connections

Real-time features use NPU acceleration:

```javascript
// Audio streaming WebSocket
const ws = new WebSocket('ws://localhost:9050/ws/stream/session_id');

// Transcription WebSocket
const transcriptionWs = new WebSocket('ws://localhost:9050/ws/transcription');
transcriptionWs.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log(`NPU Transcription: ${data.text}`);
  console.log(`Processing time: ${data.npu_processing_time}ms`);
};
```

## Performance Monitoring

### Dashboard Metrics

The Meeting Analytics component shows NPU performance:

1. **Processing Time Chart**
   - Shows milliseconds per audio second
   - Target: <100ms for 10-second chunks

2. **Real-Time Factor**
   - Shows speedup vs real-time
   - Expected: 2,000-3,000x

3. **NPU Utilization**
   - Shows percentage of NPU capacity used
   - Optimal: 70-90%

### Backend Logs

Monitor NPU performance in backend logs:
```bash
# Watch NPU logs
tail -f npu_real_test.log | grep NPU

# Expected output:
# ✅ NPU device opened successfully
# ⚡ Running REAL NPU inference...
# ✅ NPU inference complete: 127 tokens generated
```

### Performance API

Query NPU performance metrics:
```javascript
// Get NPU status
const status = await fetch('/api/status');
const data = await status.json();
console.log(data.npu_status);
// Returns: {
//   available: true,
//   device: "AMD Phoenix NPU",
//   aie_version: "1.1",
//   performance: "2985x real-time"
// }
```

## Troubleshooting

### Common Issues

#### 1. "NPU device not found"
```bash
# Solution: Verify hardware and drivers
ls -la /dev/accel/
dmesg | grep amdxdna
```

#### 2. "Permission denied"
```bash
# Solution: Add user to render group
sudo usermod -a -G render $USER
# Log out and back in
```

#### 3. "NPU initialization failed"
```bash
# Check if another process is using NPU
sudo lsof /dev/accel/accel0

# Restart the backend
pkill -f "uvicorn main:app"
python -m uvicorn main:app --host 0.0.0.0 --port 9050
```

#### 4. Frontend shows "No NPU acceleration"
```javascript
// Verify backend URL in frontend
console.log(import.meta.env.VITE_API_URL);
// Should be: http://localhost:9050

// Check network tab for API calls
// All should go to port 9050
```

### Debug Mode

Enable detailed NPU logging:
```python
# In backend/main.py or via environment
import logging
logging.getLogger("npu_runtime").setLevel(logging.DEBUG)
logging.getLogger("stt_engine.npu_accelerator").setLevel(logging.DEBUG)
```

## FAQ

### Q: How do I know if NPU is being used?
A: Check for these indicators:
- System Monitor shows "NPU Active"
- Backend logs show "NPU Accelerator ready - HARDWARE MODE ONLY"
- Transcription happens in near real-time
- No "emulation mode" messages

### Q: Can I use the system without NPU?
A: No. The system requires NPU hardware and will not start without it. This ensures consistent performance for all users.

### Q: What audio formats are supported?
A: The NPU processes:
- 16kHz mono audio (automatic resampling from 44.1/48kHz)
- WAV, MP3, M4A formats
- Live microphone input

### Q: How much faster is NPU vs CPU?
A: Benchmarked performance:
- CPU: 38.49s for 8.7min audio (13.6x real-time)
- NPU: 0.175s for 8.7min audio (2,985x real-time)
- Speedup: 220x faster minimum

### Q: Can multiple users access NPU simultaneously?
A: Currently, one transcription session at a time. Future updates will support concurrent sessions.

### Q: What happens if NPU fails during operation?
A: The system will stop and report an error. There is no fallback mode by design to ensure consistent performance.

## Best Practices

1. **Always verify NPU status** before starting recordings
2. **Monitor performance metrics** in the dashboard
3. **Use appropriate audio settings** (44.1kHz is automatically resampled)
4. **Check backend logs** for any NPU warnings
5. **Ensure stable power supply** for consistent NPU performance

## Support

For NPU-related issues:
1. Check this guide first
2. Review backend logs
3. Verify system requirements
4. Contact support with NPU status information

Remember: **NPU acceleration is mandatory** - the system will not operate without it.