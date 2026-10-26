pub mod camera;
pub mod compositor;
pub mod draw;
pub mod gui;
pub mod inference;
pub mod mailbox;
pub mod sticker;
pub mod types;

use mailbox::Mailbox;
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use std::env;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use types::{Landmarks, RgbFrame, RgbaFrame, Stats};

fn get_models_dir() -> PathBuf {
    if let Ok(dir) = env::var("AR_MODELS_DIR") {
        return PathBuf::from(dir);
    }

    // 1. Check relative to current working directory (e.g. running from repo root)
    let cwd_assets = PathBuf::from("assets/models");
    if cwd_assets.exists() {
        return cwd_assets;
    }

    // 2. Check relative to the executable location (e.g. double-clicking inside target/release or elsewhere)
    if let Ok(mut exe_dir) = env::current_exe() {
        exe_dir.pop();
        // Check exe_dir/assets/models
        let next_to_exe = exe_dir.join("assets").join("models");
        if next_to_exe.exists() {
            return next_to_exe;
        }
        // Check ../../assets/models (when running from target/release/)
        let repo_root_relative = exe_dir.join("..").join("..").join("assets").join("models");
        if repo_root_relative.exists() {
            return repo_root_relative;
        }
    }

    cwd_assets
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
    let detector_path = check_model_file(&models_dir, "face_detection_short_range.onnx")?;
    let mesh_path = check_model_file(&models_dir, "face_mesh_Nx3x192x192.onnx")?;

    let shutdown = Arc::new(AtomicBool::new(false));
    let stats = Stats::new();

    let camera_to_inference = Mailbox::<Arc<RgbFrame>>::new();
    let camera_to_compositor = Mailbox::<Arc<RgbFrame>>::new();
    let inference_to_compositor = Mailbox::<Landmarks>::new();
    let compositor_to_gui = Mailbox::<RgbaFrame>::new();

    let inference_handle = inference::spawn(
        &detector_path,
        &mesh_path,
        Arc::clone(&camera_to_inference),
        Arc::clone(&inference_to_compositor),
        Arc::clone(&stats),
        Arc::clone(&shutdown),
    )?;

    let camera_res = camera::start(
        vec![
            Arc::clone(&camera_to_inference),
            Arc::clone(&camera_to_compositor),
        ],
        Arc::clone(&stats),
    );

    let mut camera_error = None;
    let _camera = match camera_res {
        Ok(cam) => {
            log::info!("Webcam initialized successfully.");
            Some(cam)
        }
        Err(e) => {
            log::error!("Camera not available: {e}");
            camera_error = Some(e.to_string());
            None
        }
    };

    let compositor_handle = compositor::spawn(
        Arc::clone(&camera_to_compositor),
        Arc::clone(&inference_to_compositor),
        Arc::clone(&compositor_to_gui),
        Arc::clone(&shutdown),
    );

    let gui_flags = gui::Flags {
        frame_in: Arc::clone(&compositor_to_gui),
        stats: Arc::clone(&stats),
        shutdown: Arc::clone(&shutdown),
        camera_error,
    };

    // Run the iced GUI (blocks until the window is closed)
    if let Err(e) = gui::run(gui_flags) {
        log::error!("GUI error: {e}");
    }

    // Clean shutdown sequence
    shutdown.store(true, Ordering::Relaxed);
    camera_to_inference.close();
    camera_to_compositor.close();
    inference_to_compositor.close();
    compositor_to_gui.close();

    let _ = compositor_handle.join();
    let _ = inference_handle.join();
    log::info!("Shutdown complete.");

    Ok(())
}
