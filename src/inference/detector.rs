use ndarray::Array4;

#[derive(Debug, Clone, Copy)]
pub struct Anchor {
    pub cx: f32,
    pub cy: f32,
    pub w: f32,
    pub h: f32,
}

#[derive(Debug, Clone)]
pub struct Detection {
    pub cx: f32,
    pub cy: f32,
    pub w: f32,
    pub h: f32,
    pub right_eye: [f32; 2],
    pub left_eye: [f32; 2],
    pub score: f32,
}

pub struct LetterboxInfo {
    pub scale: f32,
    pub pad_x: f32,
    pub pad_y: f32,
}

/// Generates the standard 896 SSD anchors for MediaPipe BlazeFace (128x128 input).
pub fn generate_anchors() -> Vec<Anchor> {
    let mut anchors = Vec::with_capacity(896);

    // Stride 8: 16x16 grid, 2 anchors per cell -> 512 anchors
    for y in 0..16 {
        for x in 0..16 {
            let cx = (x as f32 + 0.5) / 16.0;
            let cy = (y as f32 + 0.5) / 16.0;
            for _ in 0..2 {
                anchors.push(Anchor {
                    cx,
                    cy,
                    w: 1.0,
                    h: 1.0,
                });
            }
        }
    }

    // Stride 16: 8x8 grid, 6 anchors per cell -> 384 anchors
    for y in 0..8 {
        for x in 0..8 {
            let cx = (x as f32 + 0.5) / 8.0;
            let cy = (y as f32 + 0.5) / 8.0;
            for _ in 0..6 {
                anchors.push(Anchor {
                    cx,
                    cy,
                    w: 1.0,
                    h: 1.0,
                });
            }
        }
    }

    anchors
}

#[inline]
pub fn sigmoid(x: f32) -> f32 {
    let clamped = x.clamp(-100.0, 100.0);
    1.0 / (1.0 + (-clamped).exp())
}

/// Letterboxes an RGB frame into a [1, 3, 128, 128] NCHW tensor normalized to [-1.0, 1.0].
pub fn letterbox_frame(rgb: &[u8], w: u32, h: u32) -> (Array4<f32>, LetterboxInfo) {
    let max_dim = (w.max(h)) as f32;
    let scale = 128.0 / max_dim;
    let new_w = (w as f32 * scale).round() as u32;
    let new_h = (h as f32 * scale).round() as u32;

    let pad_x = ((128.0 - new_w as f32) / 2.0).floor();
    let pad_y = ((128.0 - new_h as f32) / 2.0).floor();

    let mut tensor = Array4::<f32>::zeros((1, 3, 128, 128));

    for y in 0..new_h {
        let orig_y = ((y as f32 + 0.5) / scale).floor() as u32;
        let orig_y = orig_y.min(h - 1);
        let dst_y = (pad_y as u32 + y) as usize;

        for x in 0..new_w {
            let orig_x = ((x as f32 + 0.5) / scale).floor() as u32;
            let orig_x = orig_x.min(w - 1);
            let dst_x = (pad_x as u32 + x) as usize;

            let orig_idx = ((orig_y * w + orig_x) * 3) as usize;
            let r = rgb[orig_idx] as f32 / 127.5 - 1.0;
            let g = rgb[orig_idx + 1] as f32 / 127.5 - 1.0;
            let b = rgb[orig_idx + 2] as f32 / 127.5 - 1.0;

            tensor[[0, 0, dst_y, dst_x]] = r;
            tensor[[0, 1, dst_y, dst_x]] = g;
            tensor[[0, 2, dst_y, dst_x]] = b;
        }
    }

    (
        tensor,
        LetterboxInfo {
            scale,
            pad_x,
            pad_y,
        },
    )
}

/// Decodes raw BlazeFace model outputs (regressors [896, 16], scores [896, 1]) into detections.
pub fn decode_detections(
    regressors: &[f32],
    scores: &[f32],
    anchors: &[Anchor],
    info: &LetterboxInfo,
    score_thresh: f32,
) -> Option<Detection> {
    if regressors.len() < 896 * 16 || scores.len() < 896 || anchors.len() < 896 {
        return None;
    }

    let mut best_score = score_thresh;
    let mut best_idx = None;

    for (i, &raw_score) in scores.iter().enumerate().take(896) {
        let score = sigmoid(raw_score);
        if score > best_score {
            best_score = score;
            best_idx = Some(i);
        }
    }

    let idx = best_idx?;
    let a = &anchors[idx];
    let r_offset = idx * 16;
    let r = &regressors[r_offset..r_offset + 16];

    // Decode in 128x128 space
    let cx_norm = a.cx + r[0] / 128.0;
    let cy_norm = a.cy + r[1] / 128.0;
    let w_norm = r[2] / 128.0;
    let h_norm = r[3] / 128.0;

    // Keypoints: kp0 = right eye, kp1 = left eye
    let right_eye_x_norm = a.cx + r[4] / 128.0;
    let right_eye_y_norm = a.cy + r[5] / 128.0;
    let left_eye_x_norm = a.cx + r[6] / 128.0;
    let left_eye_y_norm = a.cy + r[7] / 128.0;

    // Convert back to original unletterboxed camera frame pixels:
    // x = (norm_x * 128 - pad_x) / scale
    let unproject_x = |nx: f32| (nx * 128.0 - info.pad_x) / info.scale;
    let unproject_y = |ny: f32| (ny * 128.0 - info.pad_y) / info.scale;

    let cx = unproject_x(cx_norm);
    let cy = unproject_y(cy_norm);
    let w = (w_norm * 128.0) / info.scale;
    let h = (h_norm * 128.0) / info.scale;

    let right_eye = [unproject_x(right_eye_x_norm), unproject_y(right_eye_y_norm)];
    let left_eye = [unproject_x(left_eye_x_norm), unproject_y(left_eye_y_norm)];

    Some(Detection {
        cx,
        cy,
        w,
        h,
        right_eye,
        left_eye,
        score: best_score,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_anchor_generation() {
        let anchors = generate_anchors();
        assert_eq!(anchors.len(), 896);

        // First anchor center at (0.5/16, 0.5/16)
        let expected_cx = 0.5 / 16.0;
        let expected_cy = 0.5 / 16.0;
        assert!((anchors[0].cx - expected_cx).abs() < 1e-6);
        assert!((anchors[0].cy - expected_cy).abs() < 1e-6);
    }

    #[test]
    fn test_decode_zero_regressor() {
        let anchors = generate_anchors();
        let regressors = vec![0.0f32; 896 * 16];
        let mut scores = vec![-10.0f32; 896];
        scores[0] = 5.0; // High score for anchor 0

        let info = LetterboxInfo {
            scale: 1.0,
            pad_x: 0.0,
            pad_y: 0.0,
        };

        let det = decode_detections(&regressors, &scores, &anchors, &info, 0.5).unwrap();
        // With zero regressors, cx and cy must equal the anchor center * 128
        assert!((det.cx - anchors[0].cx * 128.0).abs() < 1e-5);
        assert!((det.cy - anchors[0].cy * 128.0).abs() < 1e-5);
        assert!((det.right_eye[0] - anchors[0].cx * 128.0).abs() < 1e-5);
        assert!((det.left_eye[0] - anchors[0].cx * 128.0).abs() < 1e-5);
    }
}
