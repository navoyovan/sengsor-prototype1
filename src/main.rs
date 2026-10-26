pub mod camera;
pub mod compositor;
pub mod mailbox;
pub mod types;

use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use std::env;
use std::path::{Path, PathBuf};

fn get_models_dir() -> PathBuf {
    env::var("AR_MODELS_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("assets/models"))
}

fn check_model_file(dir: &Path, filename: &str) -> anyhow::Result<PathBuf> {
    let p = dir.join(filename);
    if !p.exists() {
        anyhow::bail!(
            "Model file missing: {}. Run .\\download_models.ps1",
            p.display()
        );
    }
    Ok(p)
}

fn inspect_models() -> anyhow::Result<()> {
    let models_dir = get_models_dir();
    let detector_path = check_model_file(&models_dir, "face_detection_short_range.onnx")?;
    let mesh_path = check_model_file(&models_dir, "face_mesh_Nx3x192x192.onnx")?;

    println!("=== Model Inspector ===");
    for (name, path) in [
        ("BlazeFace Detector", detector_path),
        ("Face Mesh", mesh_path),
    ] {
        println!("\nModel: {} ({})", name, path.display());
        let session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(2)?
            .commit_from_file(path)?;

        println!("Inputs:");
        for (i, input) in session.inputs.iter().enumerate() {
            println!(
                "  [{}] Name: {}, Type: {:?}",
                i, input.name, input.input_type
            );
        }

        println!("Outputs:");
        for (i, output) in session.outputs.iter().enumerate() {
            println!(
                "  [{}] Name: {}, Type: {:?}",
                i, output.name, output.output_type
            );
        }
    }

    Ok(())
}

fn main() -> anyhow::Result<()> {
    env_logger::init();

    let args: Vec<String> = env::args().collect();
    if args.iter().any(|arg| arg == "--inspect") {
        inspect_models()?;
        return Ok(());
    }

    let models_dir = get_models_dir();
    let _ = check_model_file(&models_dir, "face_detection_short_range.onnx")?;
    let _ = check_model_file(&models_dir, "face_mesh_Nx3x192x192.onnx")?;

    Ok(())
}
