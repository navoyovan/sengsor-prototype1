use crate::mailbox::Mailbox;
use crate::types::RgbaFrame;
use iced::widget::image::Handle;
use iced::widget::{container, image, text};
use iced::{
    executor, time, Application, Command, ContentFit, Element, Length, Settings, Subscription,
    Theme,
};
use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use std::time::Duration;

pub struct Flags {
    pub frame_in: Arc<Mailbox<RgbaFrame>>,
    pub shutdown: Arc<AtomicBool>,
    pub camera_error: Option<String>,
}

pub struct App {
    frame_in: Arc<Mailbox<RgbaFrame>>,
    _shutdown: Arc<AtomicBool>,
    camera_error: Option<String>,
    handle: Option<Handle>,
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
        (
            Self {
                frame_in: flags.frame_in,
                _shutdown: flags.shutdown,
                camera_error: flags.camera_error,
                handle: None,
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
                }
            }
        }
        Command::none()
    }

    fn view(&self) -> Element<'_, Self::Message> {
        let content: Element<'_, Self::Message> = match &self.handle {
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

        container(content)
            .width(Length::Fill)
            .height(Length::Fill)
            .center_x()
            .center_y()
            .into()
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
