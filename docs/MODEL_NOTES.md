# Model Notes & ONNX Inspection Results

Inspection conducted with `cargo run --release -- --inspect` using `ort = "=2.0.0-rc.10"`:

## Inspector Output

### 1. BlazeFace Detector (`face_detection_short_range.onnx`)
- **Inputs**:
  - `[0] Name: input, Type: Tensor { ty: Float32, shape: [-1, 3, 128, 128] }`
- **Outputs**:
  - `[0] Name: regressors, Type: Tensor { ty: Float32, shape: [-1, -1, -1] }` (actual: `[1, 896, 16]`)
  - `[1] Name: scores, Type: Tensor { ty: Float32, shape: [-1, -1, -1] }` (actual: `[1, 896, 1]`)

### 2. Face Mesh (`face_mesh_Nx3x192x192.onnx`)
- **Inputs**:
  - `[0] Name: input, Type: Tensor { ty: Float32, shape: [-1, 3, 192, 192] }`
- **Outputs**:
  - `[0] Name: landmarks, Type: Tensor { ty: Float32, shape: [-1, 468, 3] }`
  - `[1] Name: score, Type: Tensor { ty: Float32, shape: [-1, 1] }`

## Verified Assumptions & Pipeline Details

| Item | Specification / Verified Behavior |
|---|---|
| Detector input | RGB, `1×3×128×128` NCHW, letterboxed (keep aspect ratio, zero-pad), normalized to `[-1.0, 1.0]` via `(val / 127.5) - 1.0`. |
| Detector outputs | `regressors`: 896 anchors with 16 values each (`[cx, cy, w, h, keypoints 0..5 (x, y)]` offsets divided by 128). `scores`: 896 raw logits; filtered via `sigmoid(clamp(logit, -100, 100)) >= 0.5`. |
| Mesh input | RGB, `1×3×192×192` NCHW, normalized to `[0.0, 1.0]` via `val / 255.0`. |
| Mesh outputs | `landmarks`: 468 points `(x, y, z)` in 192-px crop space. `score`: raw face-presence logit, converted via `sigmoid(logit) >= 0.5`. |

Model files are verified present at `assets/models/`.
Overridable with `AR_MODELS_DIR` environment variable.
