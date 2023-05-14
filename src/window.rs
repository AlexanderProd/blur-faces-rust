use opencv::core::{Point, Scalar};
use opencv::imgproc;
use opencv::{highgui, prelude::*};

type Result<T> = opencv::Result<T>;

pub(crate) struct Window {
    name: String,
}

impl Window {
    pub fn create(name: &'_ str, width: i32, height: i32) -> Result<Self> {
        highgui::named_window(name, highgui::WINDOW_GUI_NORMAL | highgui::WINDOW_KEEPRATIO)?;
        highgui::resize_window(name, width, height)?;
        Ok(Self {
            name: name.to_owned(),
        })
    }

    pub fn show_image(&self, frame: &mut Mat, fps: Option<f64>) -> Result<()> {
        if let Some(fps) = fps {
            imgproc::put_text(
                frame,
                &format!("FPS: {:.2}", fps),
                Point::new(10, 20),
                imgproc::FONT_HERSHEY_SIMPLEX,
                0.5,
                Scalar::new(0., 0., 255., 0.),
                1,
                imgproc::LINE_8,
                false,
            )?;
        }

        highgui::imshow(&self.name, frame)
    }
}

impl Drop for Window {
    fn drop(&mut self) {
        let _ = highgui::destroy_window(&self.name);
    }
}
