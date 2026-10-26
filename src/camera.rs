use crate::mailbox::Mailbox;
use crate::types::RgbFrame;
use nokhwa::pixel_format::RgbFormat;
use nokhwa::utils::{CameraFormat, CameraIndex, FrameFormat, RequestedFormat, RequestedFormatType};
use nokhwa::CallbackCamera;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

pub fn start(mailboxes: Vec<Arc<Mailbox<Arc<RgbFrame>>>>) -> anyhow::Result<CallbackCamera> {
    let index = CameraIndex::Index(0);
    let requested = RequestedFormat::new::<RgbFormat>(RequestedFormatType::Closest(
        CameraFormat::new_from(640, 480, FrameFormat::MJPEG, 30),
    ));

    let frame_counter = Arc::new(AtomicU64::new(0));
    let frame_counter_cb = Arc::clone(&frame_counter);
    let mailboxes_cb = mailboxes;

    let mut camera = CallbackCamera::new(index, requested, move |buffer| {
        let image = match buffer.decode_image::<RgbFormat>() {
            Ok(img) => img,
            Err(e) => {
                log::warn!("Failed to decode camera frame: {e}");
                return;
            }
        };

        let w = image.width();
        let h = image.height();
        let data = image.into_raw();
        let id = frame_counter_cb.fetch_add(1, Ordering::Relaxed);

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
