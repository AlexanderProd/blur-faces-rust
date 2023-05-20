use crate::constants::*;
use opencv::core::Size;
use opencv::{prelude::*, videoio::VideoWriter};

type Result<T> = opencv::Result<T>;

pub(crate) struct Output {
    writer: VideoWriter,
}

impl Output {
    pub fn create(output_file: &str, width: i32, height: i32) -> Result<Self> {
        let fourcc = VideoWriter::fourcc('M', 'P', '4', 'V')?;
        let writer = VideoWriter::new(
            output_file,
            fourcc,
            OUTPUT_FPS,
            Size::new(width, height),
            true,
        )?;
        Ok(Self { writer })
    }

    pub fn is_opened(&self) -> Result<bool> {
        VideoWriter::is_opened(&self.writer)
    }

    pub fn write_frame(&mut self, frame: &Mat) -> Result<()> {
        let _ = &self.writer.write(frame)?;
        Ok(())
    }
}

impl Drop for Output {
    fn drop(&mut self) {
        let _ = self.writer.release();
    }
}
