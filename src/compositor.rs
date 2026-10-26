use crate::draw;
use crate::mailbox::Mailbox;
use crate::types::{Landmarks, RgbFrame, RgbaFrame};
use std::env;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::Duration;

/// Exponential Moving Average filter for 3 tracked 2D landmark points.
pub struct Smoother {
    alpha: f32,
    state: Option<[[f32; 2]; 3]>,
}

impl Smoother {
    pub fn new(alpha: f32) -> Self {
        Self {
            alpha: alpha.clamp(0.01, 1.0),
            state: None,
        }
    }

    pub fn reset(&mut self) {
        self.state = None;
    }

    pub fn update(&mut self, points: [[f32; 2]; 3]) -> [[f32; 2]; 3] {
        match self.state {
            None => {
                self.state = Some(points);
                points
            }
            Some(prev) => {
                let mut smoothed = [[0.0, 0.0]; 3];
                for i in 0..3 {
                    smoothed[i][0] = prev[i][0] + self.alpha * (points[i][0] - prev[i][0]);
                    smoothed[i][1] = prev[i][1] + self.alpha * (points[i][1] - prev[i][1]);
                }
                self.state = Some(smoothed);
                smoothed
            }
        }
    }
}

/// Converts an RGB8 image buffer into mirrored RGBA8 with alpha = 255 in a single pass.
pub fn mirror_rgb_to_rgba(rgb: &[u8], w: u32, h: u32, out: &mut Vec<u8>) {
    let out_len = (w * h * 4) as usize;
    if out.len() != out_len {
        out.resize(out_len, 255);
    }

    let w_usize = w as usize;
    let h_usize = h as usize;

    for y in 0..h_usize {
        let src_row_start = y * w_usize * 3;
        let dst_row_start = y * w_usize * 4;

        for x in 0..w_usize {
            let src_x = x;
            let dst_x = w_usize - 1 - x;

            let src_idx = src_row_start + src_x * 3;
            let dst_idx = dst_row_start + dst_x * 4;

            out[dst_idx] = rgb[src_idx];
            out[dst_idx + 1] = rgb[src_idx + 1];
            out[dst_idx + 2] = rgb[src_idx + 2];
            out[dst_idx + 3] = 255;
        }
    }
}

pub fn spawn(
    frames_in: Arc<Mailbox<Arc<RgbFrame>>>,
    landmarks_in: Arc<Mailbox<Landmarks>>,
    out: Arc<Mailbox<RgbaFrame>>,
    shutdown: Arc<AtomicBool>,
) -> JoinHandle<()> {
    let debug_mode = env::var("AR_DEBUG").map(|v| v == "1").unwrap_or(false);
    let alpha = env::var("AR_EMA")
        .ok()
        .and_then(|v| v.parse::<f32>().ok())
        .unwrap_or(0.35);

    thread::spawn(move || {
        let mut buf = Vec::new();
        let mut _smoother = Smoother::new(alpha);

        while !shutdown.load(Ordering::Relaxed) {
            let frame = match frames_in.take_blocking() {
                Some(f) => f,
                None => break,
            };

            mirror_rgb_to_rgba(&frame.data, frame.w, frame.h, &mut buf);

            if let Some(lm) = landmarks_in.get() {
                if lm.at.elapsed() < Duration::from_millis(250) && debug_mode {
                    for (i, pt) in lm.points.iter().enumerate() {
                        if i % 4 == 0 {
                            let mirrored_x = frame.w as f32 - 1.0 - pt[0];
                            draw::dot(
                                &mut buf,
                                frame.w,
                                frame.h,
                                mirrored_x,
                                pt[1],
                                2,
                                [0, 255, 120, 255],
                            );
                        }
                    }
                }
            }

            out.put(RgbaFrame {
                w: frame.w,
                h: frame.h,
                data: buf.clone(),
            });
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_smoother_convergence_and_reset() {
        let mut s = Smoother::new(0.5);
        let pt1 = [[10.0, 10.0], [20.0, 20.0], [30.0, 30.0]];
        let r1 = s.update(pt1);
        assert_eq!(r1, pt1);

        // Feed constant input pt2
        let pt2 = [[20.0, 20.0], [40.0, 40.0], [60.0, 60.0]];
        let r2 = s.update(pt2);
        // Halfway between 10 and 20 is 15
        assert!((r2[0][0] - 15.0).abs() < 1e-4);

        // Reset
        s.reset();
        assert!(s.state.is_none());
        let r3 = s.update(pt2);
        // After reset, first update takes target directly
        assert_eq!(r3, pt2);
    }

    #[test]
    fn test_mirror_rgb_to_rgba_3x2() {
        let rgb = vec![
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
        ];
        let mut out = Vec::new();
        mirror_rgb_to_rgba(&rgb, 3, 2, &mut out);

        let expected = vec![
            7, 8, 9, 255, 4, 5, 6, 255, 1, 2, 3, 255, 16, 17, 18, 255, 13, 14, 15, 255, 10, 11, 12,
            255,
        ];
        assert_eq!(out, expected);
    }
}
