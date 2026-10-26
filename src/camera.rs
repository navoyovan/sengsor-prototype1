use crate::mailbox::Mailbox;
use crate::types::{RgbFrame, Stats};
use nokhwa::pixel_format::RgbFormat;
use nokhwa::utils::{CameraIndex, RequestedFormat, RequestedFormatType};
use nokhwa::CallbackCamera;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

pub fn start(
    mailboxes: Vec<Arc<Mailbox<Arc<RgbFrame>>>>,
    stats: Arc<Stats>,
) -> anyhow::Result<CallbackCamera> {
    let index = CameraIndex::Index(0);
    // Request highest frame rate / closest format without hard-pinning MJPEG (which some cameras don't support in MediaFoundation)
    let requested = RequestedFormat::new::<RgbFormat>(RequestedFormatType::HighestFrameRate(30));

    let frame_counter = Arc::new(AtomicU64::new(0));
    let frame_counter_cb = Arc::clone(&frame_counter);
    let mailboxes_cb = mailboxes;

    let mut last_fps_time = Instant::now();
    let mut fps_counter = 0u32;

    let mut camera = CallbackCamera::new(index, requested, move |buffer| {
        let image = match buffer.decode_image::<RgbFormat>() {
            Ok(img) => img,
            Err(e) => {
                log::warn!("Failed to decode camera frame: {e}");
                return;
            }
        };

        fps_counter += 1;
        let now = Instant::now();
        if now.duration_since(last_fps_time).as_secs_f32() >= 1.0 {
            stats.set_cam_fps(fps_counter);
            fps_counter = 0;
            last_fps_time = now;
        }

        let w = image.width();
        let h = image.height();
        let data = image.into_raw();
        let id = frame_counter_cb.fetch_add(1, Ordering::Relaxed);
        stats.cam_frame_count.store(id, Ordering::Relaxed);

        let frame = Arc::new(RgbFrame { id, w, h, data });
        for mb in &mailboxes_cb {
            mb.put(Arc::clone(&frame));
        }
    })?;

    camera.open_stream()?;
    log::info!(
        "Camera stream opened successfully. Format negotiated: {:?}",
        camera.camera_format()
    );

    Ok(camera)
}
