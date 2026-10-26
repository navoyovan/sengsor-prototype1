use crate::draw;
use crate::mailbox::Mailbox;
use crate::sticker::{transform_crown, Transform};
use crate::types::{Landmarks, RgbFrame, RgbaFrame, Stats};
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
    stats: Arc<Stats>,
    shutdown: Arc<AtomicBool>,
) -> JoinHandle<()> {
    let debug_mode_env = env::var("AR_DEBUG").map(|v| v == "1").unwrap_or(false);
    if debug_mode_env {
        stats.debug_dots.store(true, Ordering::Relaxed);
    }
    if let Ok(var_val) = env::var("AR_EMA") {
        if let Ok(val) = var_val.parse::<f32>() {
            stats
                .ema_alpha_pct
                .store((val * 100.0).clamp(1.0, 100.0) as u32, Ordering::Relaxed);
        }
    }

    thread::spawn(move || {
        let mut buf = Vec::new();
        let mut smoother = Smoother::new(stats.ema_alpha());

        while !shutdown.load(Ordering::Relaxed) {
            let frame = match frames_in.take_blocking() {
                Some(f) => f,
                None => break,
            };

            mirror_rgb_to_rgba(&frame.data, frame.w, frame.h, &mut buf);

            let mut face_active = false;
            let current_alpha = stats.ema_alpha();
            smoother.alpha = current_alpha.clamp(0.01, 1.0);

            let show_dots = stats.debug_dots.load(Ordering::Relaxed);
            let show_crown = stats.draw_crown.load(Ordering::Relaxed);

            if let Some(lm) = landmarks_in.get() {
                if lm.at.elapsed() < Duration::from_millis(250) && lm.points.len() >= 468 {
                    face_active = true;

                    let pt_33 = lm.points[33];
                    let pt_263 = lm.points[263];
                    let pt_10 = lm.points[10];

                    let mirror_x = |x: f32| frame.w as f32 - 1.0 - x;

                    let raw_points = [
                        [mirror_x(pt_33[0]), pt_33[1]],
                        [mirror_x(pt_263[0]), pt_263[1]],
                        [mirror_x(pt_10[0]), pt_10[1]],
                    ];

                    let smoothed = smoother.update(raw_points);

                    if show_crown {
                        let eye_a = smoothed[0];
                        let eye_b = smoothed[1];
                        let anchor = smoothed[2];

                        let transform = Transform::from_screen_points(eye_a, eye_b, anchor);
                        let segments = transform_crown(&transform);

                        for (p1, p2) in segments {
                            draw::line(&mut buf, frame.w, frame.h, p1, p2, 3, [0, 255, 120, 255]);
                        }
                    }

                    if show_dots {
                        for (i, pt) in lm.points.iter().enumerate() {
                            if i % 4 == 0 {
                                let mirrored_x = mirror_x(pt[0]);
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
            }

            if !face_active {
                smoother.reset();
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

        let pt2 = [[20.0, 20.0], [40.0, 40.0], [60.0, 60.0]];
        let r2 = s.update(pt2);
        assert!((r2[0][0] - 15.0).abs() < 1e-4);

        s.reset();
        assert!(s.state.is_none());
        let r3 = s.update(pt2);
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
