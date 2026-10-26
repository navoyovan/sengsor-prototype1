use crate::inference::detector::Detection;
use ndarray::Array4;

#[derive(Debug, Clone, Copy)]
pub struct Roi {
    pub cx: f32,
    pub cy: f32,
    pub side: f32,
    pub theta: f32, // in radians
}

impl Roi {
    pub fn from_detection(det: &Detection) -> Self {
        let side = det.w.max(det.h) * 1.5;
        let ex = det.left_eye[0] - det.right_eye[0];
        let ey = det.left_eye[1] - det.right_eye[1];
        let theta = ey.atan2(ex);

        Self {
            cx: det.cx,
            cy: det.cy,
            side,
            theta,
        }
    }

    /// Back-project a point (px, py) from 192x192 crop space to original unmirrored frame coordinates.
    pub fn back_project(&self, px: f32, py: f32) -> [f32; 2] {
        let u = px / 192.0 - 0.5;
        let v = py / 192.0 - 0.5;

        let cos_t = self.theta.cos();
        let sin_t = self.theta.sin();

        let rx = cos_t * (u * self.side) - sin_t * (v * self.side);
        let ry = sin_t * (u * self.side) + cos_t * (v * self.side);

        [self.cx + rx, self.cy + ry]
    }
}

/// Bilinear sampling helper for an RGB frame buffer. Returns 0 if out of bounds.
#[inline]
fn sample_bilinear_rgb(rgb: &[u8], w: u32, h: u32, sx: f32, sy: f32) -> [f32; 3] {
    if sx < 0.0 || sy < 0.0 || sx >= (w as f32 - 1.0) || sy >= (h as f32 - 1.0) {
        if sx >= 0.0 && sx < w as f32 && sy >= 0.0 && sy < h as f32 {
            let x0 = sx.floor() as u32;
            let y0 = sy.floor() as u32;
            let idx = ((y0 * w + x0) * 3) as usize;
            return [
                rgb[idx] as f32 / 255.0,
                rgb[idx + 1] as f32 / 255.0,
                rgb[idx + 2] as f32 / 255.0,
            ];
        }
        return [0.0, 0.0, 0.0];
    }

    let x0 = sx.floor() as u32;
    let y0 = sy.floor() as u32;
    let x1 = x0 + 1;
    let y1 = y0 + 1;

    let fx = sx - x0 as f32;
    let fy = sy - y0 as f32;

    let idx00 = ((y0 * w + x0) * 3) as usize;
    let idx10 = ((y0 * w + x1) * 3) as usize;
    let idx01 = ((y1 * w + x0) * 3) as usize;
    let idx11 = ((y1 * w + x1) * 3) as usize;

    let mut out = [0.0f32; 3];
    for c in 0..3 {
        let p00 = rgb[idx00 + c] as f32;
        let p10 = rgb[idx10 + c] as f32;
        let p01 = rgb[idx01 + c] as f32;
        let p11 = rgb[idx11 + c] as f32;

        let top = p00 * (1.0 - fx) + p10 * fx;
        let bot = p01 * (1.0 - fx) + p11 * fx;
        out[c] = (top * (1.0 - fy) + bot * fy) / 255.0;
    }

    out
}

/// Samples a rotated square ROI of side S into a 1x3x192x192 tensor normalized to [0.0, 1.0].
pub fn sample_rotated_roi(rgb: &[u8], w: u32, h: u32, roi: &Roi) -> Array4<f32> {
    let mut tensor = Array4::<f32>::zeros((1, 3, 192, 192));
    let cos_t = roi.theta.cos();
    let sin_t = roi.theta.sin();

    for j in 0..192 {
        let v = (j as f32 + 0.5) / 192.0 - 0.5;
        for i in 0..192 {
            let u = (i as f32 + 0.5) / 192.0 - 0.5;

            let rx = cos_t * (u * roi.side) - sin_t * (v * roi.side);
            let ry = sin_t * (u * roi.side) + cos_t * (v * roi.side);

            let src_x = roi.cx + rx;
            let src_y = roi.cy + ry;

            let sampled = sample_bilinear_rgb(rgb, w, h, src_x, src_y);
            tensor[[0, 0, j, i]] = sampled[0];
            tensor[[0, 1, j, i]] = sampled[1];
            tensor[[0, 2, j, i]] = sampled[2];
        }
    }

    tensor
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    #[test]
    fn test_back_project_center() {
        let roi = Roi {
            cx: 320.0,
            cy: 240.0,
            side: 100.0,
            theta: 0.5,
        };
        // In 192x192 space, crop center is (96.0, 96.0)
        let pt = roi.back_project(96.0, 96.0);
        assert!((pt[0] - 320.0).abs() < 1e-4);
        assert!((pt[1] - 240.0).abs() < 1e-4);
    }

    #[test]
    fn test_back_project_rotated_90_deg() {
        let roi = Roi {
            cx: 0.0,
            cy: 0.0,
            side: 192.0,
            theta: PI / 2.0, // 90 degrees: cos=0, sin=1
        };
        // A point at (+10, 0) relative to center: px = 96 + 10 = 106, py = 96
        // u = 10/192, v = 0 -> R(90) * (u*S, 0) = (0, u*S) = (0, 10)
        let pt = roi.back_project(106.0, 96.0);
        assert!((pt[0] - 0.0).abs() < 1e-3);
        assert!((pt[1] - 10.0).abs() < 1e-3);
    }

    #[test]
    fn test_synthetic_gradient_sampling() {
        let w = 200;
        let h = 200;
        let mut rgb = vec![0u8; (w * h * 3) as usize];
        for y in 0..h {
            for x in 0..w {
                let idx = ((y * w + x) * 3) as usize;
                rgb[idx] = x as u8; // R = x
                rgb[idx + 1] = y as u8; // G = y
                rgb[idx + 2] = 128; // B = 128
            }
        }

        let roi = Roi {
            cx: 100.0,
            cy: 100.0,
            side: 192.0,
            theta: 0.0,
        };
        let tensor = sample_rotated_roi(&rgb, w, h, &roi);
        // Center pixel (96, 96) should correspond to src (100.0, 100.0) -> R = 100/255
        let center_r = tensor[[0, 0, 96, 96]];
        let expected_r = 100.0 / 255.0;
        assert!((center_r - expected_r).abs() < 0.01);
    }
}
