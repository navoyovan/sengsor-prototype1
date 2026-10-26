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
