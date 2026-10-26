use crate::camera::AvailableCamera;
use crate::mailbox::Mailbox;
use crate::types::{RgbaFrame, Stats};
use iced::widget::image::Handle;
use iced::widget::{button, checkbox, column, container, image, pick_list, row, slider, text};
use iced::{
    executor, time, Alignment, Application, Command, ContentFit, Element, Length, Settings,
    Subscription, Theme,
};
use std::fmt;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CameraChoice {
    pub index_num: u32,
    pub name: String,
}

impl fmt::Display for CameraChoice {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}] {}", self.index_num, self.name)
    }
}

pub struct Flags {
    pub frame_in: Arc<Mailbox<RgbaFrame>>,
    pub stats: Arc<Stats>,
    pub shutdown: Arc<AtomicBool>,
    pub camera_error: Option<String>,
    pub available_cameras: Vec<AvailableCamera>,
}

pub struct App {
    frame_in: Arc<Mailbox<RgbaFrame>>,
    stats: Arc<Stats>,
    _shutdown: Arc<AtomicBool>,
    camera_error: Option<String>,
    handle: Option<Handle>,
    show_debug_menu: bool,
    last_frame_time: Instant,
    display_frame_counter: u32,
    hud_cam_fps: u32,
    hud_disp_fps: u32,
    hud_infer_ms: u32,
    hud_face: bool,
    // Debug state cached for UI
    debug_dots: bool,
    draw_crown: bool,
    fast_tracking: bool,
    ema_alpha_pct: u32,
    // Camera list & selection
    camera_options: Vec<CameraChoice>,
    selected_camera: Option<CameraChoice>,
}

#[derive(Debug, Clone)]
pub enum Message {
    Tick,
    ToggleDebugMenu,
    ToggleDebugDots(bool),
    ToggleDrawCrown(bool),
    ToggleFastTracking(bool),
    ChangeEmaAlpha(u32),
    SelectCamera(CameraChoice),
}

impl Application for App {
    type Executor = executor::Default;
    type Message = Message;
    type Theme = Theme;
    type Flags = Flags;

    fn new(flags: Self::Flags) -> (Self, Command<Self::Message>) {
        let debug_dots = flags.stats.debug_dots.load(Ordering::Relaxed);
        let draw_crown = flags.stats.draw_crown.load(Ordering::Relaxed);
        let fast_tracking = flags.stats.skip_detection_tracking.load(Ordering::Relaxed);
        let ema_alpha_pct = flags.stats.ema_alpha_pct.load(Ordering::Relaxed);

        let camera_options: Vec<CameraChoice> = flags
            .available_cameras
            .into_iter()
            .map(|c| CameraChoice {
                index_num: c.index_num,
                name: c.name,
            })
            .collect();

        let current_index = flags.stats.selected_camera_index.load(Ordering::Relaxed);
        let selected_camera = camera_options
            .iter()
            .find(|c| c.index_num == current_index)
            .cloned()
            .or_else(|| camera_options.first().cloned());

        (
            Self {
                frame_in: flags.frame_in,
                stats: flags.stats,
                _shutdown: flags.shutdown,
                camera_error: flags.camera_error,
                handle: None,
                show_debug_menu: true,
                last_frame_time: Instant::now(),
                display_frame_counter: 0,
                hud_cam_fps: 0,
                hud_disp_fps: 0,
                hud_infer_ms: 0,
                hud_face: false,
                debug_dots,
                draw_crown,
                fast_tracking,
                ema_alpha_pct,
                camera_options,
                selected_camera,
            },
            Command::none(),
        )
    }

    fn title(&self) -> String {
        "Rust AR Sticker — MediaPipe Face Tracking".to_string()
    }

    fn update(&mut self, message: Self::Message) -> Command<Self::Message> {
        match message {
            Message::Tick => {
                if let Some(f) = self.frame_in.try_take() {
                    self.handle = Some(Handle::from_pixels(f.w, f.h, f.data));
                    self.display_frame_counter += 1;
                }

                let now = Instant::now();
                if now.duration_since(self.last_frame_time).as_secs_f32() >= 1.0 {
                    self.stats.set_display_fps(self.display_frame_counter);
                    self.hud_disp_fps = self.display_frame_counter;
                    self.display_frame_counter = 0;
                    self.last_frame_time = now;

                    self.hud_cam_fps = self.stats.cam_fps.load(Ordering::Relaxed);
                    self.hud_infer_ms = self.stats.inference_ms.load(Ordering::Relaxed);
                    self.hud_face = self.stats.face_detected.load(Ordering::Relaxed);
                }
            }
            Message::ToggleDebugMenu => {
                self.show_debug_menu = !self.show_debug_menu;
            }
            Message::ToggleDebugDots(v) => {
                self.debug_dots = v;
                self.stats.debug_dots.store(v, Ordering::Relaxed);
            }
            Message::ToggleDrawCrown(v) => {
                self.draw_crown = v;
                self.stats.draw_crown.store(v, Ordering::Relaxed);
            }
            Message::ToggleFastTracking(v) => {
                self.fast_tracking = v;
                self.stats
                    .skip_detection_tracking
                    .store(v, Ordering::Relaxed);
            }
            Message::ChangeEmaAlpha(v) => {
                self.ema_alpha_pct = v;
                self.stats.ema_alpha_pct.store(v, Ordering::Relaxed);
            }
            Message::SelectCamera(choice) => {
                self.selected_camera = Some(choice.clone());
                self.stats
                    .selected_camera_index
                    .store(choice.index_num, Ordering::Relaxed);
                self.stats
                    .switch_camera_requested
                    .store(true, Ordering::Relaxed);
            }
        }
        Command::none()
    }

    fn view(&self) -> Element<'_, Self::Message> {
        let video_content: Element<'_, Self::Message> = match &self.handle {
            Some(handle) => image(handle.clone())
                .content_fit(ContentFit::Contain)
                .width(Length::Fill)
                .height(Length::Fill)
                .into(),
            None => {
                if let Some(err) = &self.camera_error {
                    text(format!("Camera not available: {err}")).size(20).into()
                } else {
                    text("Waiting for camera...").size(24).into()
                }
            }
        };

        let menu_toggle_btn = button(
            text(if self.show_debug_menu {
                "Hide Controls"
            } else {
                "Show Controls"
            })
            .size(12),
        )
        .on_press(Message::ToggleDebugMenu);

        let face_str = if self.hud_face { "YES" } else { "NO" };
        let hud_summary = text(format!(
            "Cam: {} FPS | Disp: {} FPS | Inference: {} ms | Face: {}",
            self.hud_cam_fps, self.hud_disp_fps, self.hud_infer_ms, face_str
        ))
        .size(13);

        let top_bar = row![menu_toggle_btn, hud_summary]
            .spacing(16)
            .align_items(Alignment::Center)
            .padding(8);

        if self.show_debug_menu {
            let camera_picker = row![
                text("Select Camera:").size(14),
                pick_list(
                    self.camera_options.clone(),
                    self.selected_camera.clone(),
                    Message::SelectCamera
                )
                .width(Length::Fixed(240.0))
            ]
            .spacing(8)
            .align_items(Alignment::Center);

            let dot_toggle = checkbox("Show 468 Landmark Dots", self.debug_dots)
                .on_toggle(Message::ToggleDebugDots)
                .size(15);

            let crown_toggle = checkbox("Show Crown Sticker", self.draw_crown)
                .on_toggle(Message::ToggleDrawCrown)
                .size(15);

            let tracking_toggle = checkbox("Fast Tracking", self.fast_tracking)
                .on_toggle(Message::ToggleFastTracking)
                .size(15);

            let alpha_f = self.ema_alpha_pct as f32 / 100.0;
            let alpha_label = if self.ema_alpha_pct == 100 {
                "Smoothing Alpha: 1.00 (Zero Lag)".to_string()
            } else {
                format!("Smoothing Alpha: {:.2}", alpha_f)
            };

            let alpha_slider = row![
                text(alpha_label).size(13),
                slider(5..=100, self.ema_alpha_pct, Message::ChangeEmaAlpha)
                    .width(Length::Fixed(160.0))
            ]
            .spacing(12)
            .align_items(Alignment::Center);

            let controls_panel = container(
                column![
                    row![camera_picker, alpha_slider]
                        .spacing(24)
                        .align_items(Alignment::Center),
                    row![dot_toggle, crown_toggle, tracking_toggle]
                        .spacing(20)
                        .align_items(Alignment::Center),
                ]
                .spacing(8)
                .padding(8),
            )
            .width(Length::Fill);

            column![video_content, top_bar, controls_panel]
                .width(Length::Fill)
                .height(Length::Fill)
                .into()
        } else {
            column![video_content, top_bar]
                .width(Length::Fill)
                .height(Length::Fill)
                .into()
        }
    }

    fn subscription(&self) -> Subscription<Self::Message> {
        time::every(Duration::from_millis(16)).map(|_| Message::Tick)
    }
}

pub fn run(flags: Flags) -> iced::Result {
    let settings = Settings {
        flags,
        id: None,
        window: iced::window::Settings {
            size: iced::Size::new(960.0, 720.0),
            ..Default::default()
        },
        fonts: Vec::new(),
        default_font: Default::default(),
        default_text_size: iced::Pixels(16.0),
        antialiasing: false,
    };

    App::run(settings)
}
