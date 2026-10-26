use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

pub struct RgbFrame {
    pub id: u64,
    pub w: u32,
    pub h: u32,
    pub data: Vec<u8>, // RGB8, unmirrored
}

pub struct RgbaFrame {
    pub w: u32,
    pub h: u32,
    pub data: Vec<u8>, // RGBA8, mirrored, ready for iced
}

#[derive(Clone)]
pub struct Landmarks {
    pub frame_id: u64,
    pub frame_w: u32,
    pub frame_h: u32,
    pub points: Vec<[f32; 2]>, // 468 points, unmirrored camera pixels
    pub score: f32,            // sigmoid(face-presence logit)
    pub at: Instant,
}

/// Shared real-time pipeline telemetry for the HUD overlay.
#[derive(Default)]
pub struct Stats {
    pub cam_fps: AtomicU32,
    pub display_fps: AtomicU32,
    pub inference_ms: AtomicU32,
    pub face_detected: AtomicBool,
    pub cam_frame_count: AtomicU64,
}

impl Stats {
    pub fn new() -> Arc<Self> {
        Arc::new(Self::default())
    }

    pub fn set_cam_fps(&self, fps: u32) {
        self.cam_fps.store(fps, Ordering::Relaxed);
    }

    pub fn set_display_fps(&self, fps: u32) {
        self.display_fps.store(fps, Ordering::Relaxed);
    }

    pub fn set_inference_ms(&self, ms: u32) {
        self.inference_ms.store(ms, Ordering::Relaxed);
    }

    pub fn set_face_detected(&self, detected: bool) {
        self.face_detected.store(detected, Ordering::Relaxed);
    }
}
