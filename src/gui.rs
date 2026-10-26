use crate::mailbox::Mailbox;
use crate::types::{RgbaFrame, Stats};
use iced::widget::image::Handle;
use iced::widget::{column, container, image, row, text};
use iced::{
    executor, time, Alignment, Application, Command, ContentFit, Element, Length, Settings,
    Subscription, Theme,
};
use std::env;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

pub struct Flags {
    pub frame_in: Arc<Mailbox<RgbaFrame>>,
    pub stats: Arc<Stats>,
    pub shutdown: Arc<AtomicBool>,
    pub camera_error: Option<String>,
}

pub struct App {
    frame_in: Arc<Mailbox<RgbaFrame>>,
    stats: Arc<Stats>,
    _shutdown: Arc<AtomicBool>,
    camera_error: Option<String>,
    handle: Option<Handle>,
    hud_enabled: bool,
    last_frame_time: Instant,
    display_frame_counter: u32,
    hud_cam_fps: u32,
    hud_disp_fps: u32,
    hud_infer_ms: u32,
    hud_face: bool,
}

#[derive(Debug, Clone, Copy)]
pub enum Message {
    Tick,
}

impl Application for App {
    type Executor = executor::Default;
    type Message = Message;
    type Theme = Theme;
    type Flags = Flags;

    fn new(flags: Self::Flags) -> (Self, Command<Self::Message>) {
        let hud_enabled = env::var("AR_HUD")
            .map(|v| v == "1")
            .unwrap_or(cfg!(debug_assertions));

        (
            Self {
                frame_in: flags.frame_in,
                stats: flags.stats,
                _shutdown: flags.shutdown,
                camera_error: flags.camera_error,
                handle: None,
                hud_enabled,
                last_frame_time: Instant::now(),
                display_frame_counter: 0,
                hud_cam_fps: 0,
                hud_disp_fps: 0,
                hud_infer_ms: 0,
                hud_face: false,
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

        if self.hud_enabled {
            let face_str = if self.hud_face { "YES" } else { "NO" };
            let hud_text = format!(
                "Camera: {} FPS | Display: {} FPS | Inference: {} ms | Face Tracked: {}",
                self.hud_cam_fps, self.hud_disp_fps, self.hud_infer_ms, face_str
            );

            let hud_bar = container(
                row![text(hud_text).size(14)]
                    .align_items(Alignment::Center)
                    .padding(6),
            )
            .width(Length::Fill)
            .center_x();

            column![video_content, hud_bar]
                .width(Length::Fill)
                .height(Length::Fill)
                .into()
        } else {
            container(video_content)
                .width(Length::Fill)
                .height(Length::Fill)
                .center_x()
                .center_y()
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
