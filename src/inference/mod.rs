pub mod detector;
pub mod mesh;
pub mod roi;

use crate::mailbox::Mailbox;
use crate::types::{Landmarks, RgbFrame};
use detector::{decode_detections, generate_anchors, letterbox_frame, Anchor};
use mesh::MeshSession;
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::TensorRef;
use roi::{sample_rotated_roi, Roi};
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::Instant;

pub struct DetectorSession {
    session: Session,
    anchors: Vec<Anchor>,
}

impl DetectorSession {
    pub fn new(model_path: &Path) -> anyhow::Result<Self> {
        let session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(2)?
            .commit_from_file(model_path)?;
        let anchors = generate_anchors();
        Ok(Self { session, anchors })
    }

    pub fn detect(
        &mut self,
        rgb: &[u8],
        w: u32,
        h: u32,
        score_thresh: f32,
    ) -> anyhow::Result<Option<detector::Detection>> {
        let (tensor, info) = letterbox_frame(rgb, w, h);
        let input_val = TensorRef::from_array_view(&tensor)?;
        let outputs = self.session.run(ort::inputs![input_val])?;

        let (_reg_shape, regressors) = outputs[0].try_extract_tensor::<f32>()?;
        let (_score_shape, scores) = outputs[1].try_extract_tensor::<f32>()?;

        Ok(decode_detections(
            regressors,
            scores,
            &self.anchors,
            &info,
            score_thresh,
        ))
    }
}

pub fn spawn(
    detector_path: &Path,
    mesh_path: &Path,
    frames_in: Arc<Mailbox<Arc<RgbFrame>>>,
    landmarks_out: Arc<Mailbox<Landmarks>>,
    shutdown: Arc<AtomicBool>,
) -> anyhow::Result<JoinHandle<()>> {
    let mut detector = DetectorSession::new(detector_path)?;
    let mut mesh = MeshSession::new(mesh_path)?;

    let handle = thread::spawn(move || {
        while !shutdown.load(Ordering::Relaxed) {
            let frame = match frames_in.take_blocking() {
                Some(f) => f,
                None => break,
            };

            let detection = match detector.detect(&frame.data, frame.w, frame.h, 0.5) {
                Ok(Some(d)) => d,
                Ok(None) => {
                    landmarks_out.clear();
                    continue;
                }
                Err(e) => {
                    log::warn!("Detector error: {e}");
                    landmarks_out.clear();
                    continue;
                }
            };

            let roi = Roi::from_detection(&detection);
            let roi_tensor = sample_rotated_roi(&frame.data, frame.w, frame.h, &roi);

            match mesh.run(&roi_tensor, &roi, 0.5) {
                Ok(Some(result)) => {
                    landmarks_out.put(Landmarks {
                        frame_id: frame.id,
                        frame_w: frame.w,
                        frame_h: frame.h,
                        points: result.points,
                        score: result.score,
                        at: Instant::now(),
                    });
                }
                Ok(None) => {
                    landmarks_out.clear();
                }
                Err(e) => {
                    log::warn!("Mesh error: {e}");
                    landmarks_out.clear();
                }
            }
        }
    });

    Ok(handle)
}
