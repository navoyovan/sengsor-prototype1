use crate::inference::detector::sigmoid;
use crate::inference::roi::Roi;
use ndarray::Array4;
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::TensorRef;
use std::path::Path;

pub struct MeshSession {
    session: Session,
}

pub struct MeshResult {
    pub points: Vec<[f32; 2]>, // 468 points in camera frame pixels
    pub score: f32,
}

impl MeshSession {
    pub fn new(model_path: &Path) -> anyhow::Result<Self> {
        let session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(2)?
            .commit_from_file(model_path)?;
        Ok(Self { session })
    }

    pub fn run(
        &mut self,
        tensor: &Array4<f32>,
        roi: &Roi,
        min_score: f32,
    ) -> anyhow::Result<Option<MeshResult>> {
        let input_val = TensorRef::from_array_view(tensor)?;
        let outputs = self.session.run(ort::inputs![input_val])?;

        // Output [0]: landmarks [-1, 468, 3]
        // Output [1]: score [-1, 1]
        let (_score_shape, score_data) = outputs[1].try_extract_tensor::<f32>()?;
        let logit = score_data.first().copied().unwrap_or(-100.0);
        let score = sigmoid(logit);

        if score < min_score {
            return Ok(None);
        }

        let (_lm_shape, lm_data) = outputs[0].try_extract_tensor::<f32>()?;
        if lm_data.len() < 468 * 3 {
            return Ok(None);
        }

        let mut points = Vec::with_capacity(468);
        for i in 0..468 {
            let px = lm_data[i * 3];
            let py = lm_data[i * 3 + 1];
            let frame_pt = roi.back_project(px, py);
            points.push(frame_pt);
        }

        Ok(Some(MeshResult { points, score }))
    }
}
