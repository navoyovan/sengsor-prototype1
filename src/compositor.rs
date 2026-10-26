use crate::mailbox::Mailbox;
use crate::types::{Landmarks, RgbFrame, RgbaFrame};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle};

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
    thread::spawn(move || {
        let mut buf = Vec::new();

        while !shutdown.load(Ordering::Relaxed) {
            let frame = match frames_in.take_blocking() {
                Some(f) => f,
                None => break, // Mailbox closed or shutting down
            };

            mirror_rgb_to_rgba(&frame.data, frame.w, frame.h, &mut buf);

            // Landmark drawing will be added in Phase 3/4.
            // Consume landmarks_in safely to keep it referenced:
            let _ = landmarks_in.get();

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
    fn test_mirror_rgb_to_rgba_3x2() {
        // 3x2 image:
        // Row 0: [R1, G1, B1], [R2, G2, B2], [R3, G3, B3]
        // Row 1: [R4, G4, B4], [R5, G5, B5], [R6, G6, B6]
        let rgb = vec![
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
        ];
        let mut out = Vec::new();
        mirror_rgb_to_rgba(&rgb, 3, 2, &mut out);

        // Expected mirrored:
        // Row 0: [R3, G3, B3, 255], [R2, G2, B2, 255], [R1, G1, B1, 255]
        // Row 1: [R6, G6, B6, 255], [R5, G5, B5, 255], [R4, G4, B4, 255]
        let expected = vec![
            7, 8, 9, 255, 4, 5, 6, 255, 1, 2, 3, 255, 16, 17, 18, 255, 13, 14, 15, 255, 10, 11, 12,
            255,
        ];
        assert_eq!(out, expected);
    }
}
