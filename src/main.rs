mod capture;
mod window;

use anyhow::Result;
use capture::Capture;
use opencv::core::{Mat, Rect, Scalar, Size};
use opencv::{highgui, imgproc, objdetect, prelude::*, types};
use window::Window;

const CASCADE_XML_FILE: &str = "haarcascade_frontalface_alt.xml";

const CAPTURE_WIDTH: i32 = 800;
const CAPTURE_HEIGHT: i32 = 600;

const SCALE_FACTOR: f64 = 1f64;
const SCALE_FACTOR_INV: i32 = (1f64 / SCALE_FACTOR) as i32;

const USE_BLUR: bool = true;
const BLUR_STRENGTH: i32 = 77;

const Q_KEY_CODE: i32 = 113;

fn preprocess_image(frame: &Mat) -> Result<Mat> {
    let gray = convert_to_grayscale(frame)?;
    equalize_image(&gray)
}

fn convert_to_grayscale(frame: &Mat) -> Result<Mat> {
    let mut gray = Mat::default();
    imgproc::cvt_color(frame, &mut gray, imgproc::COLOR_BGR2GRAY, 0)?;
    Ok(gray)
}

fn equalize_image(reduced: &Mat) -> Result<Mat> {
    let mut equalized = Mat::default();
    imgproc::equalize_hist(reduced, &mut equalized)?;
    Ok(equalized)
}

fn detect_faces(
    classifier: &mut objdetect::CascadeClassifier,
    image: Mat,
) -> Result<types::VectorOfRect> {
    const SCALE_FACTOR: f64 = 1.1;
    const MIN_NEIGHBORS: i32 = 2;
    const FLAGS: i32 = 0;
    const MIN_FACE_SIZE: Size = Size {
        width: 30,
        height: 30,
    };
    const MAX_FACE_SIZE: Size = Size {
        width: 0,
        height: 0,
    };

    let mut faces = types::VectorOfRect::new();
    classifier.detect_multi_scale(
        &image,
        &mut faces,
        SCALE_FACTOR,
        MIN_NEIGHBORS,
        FLAGS,
        MIN_FACE_SIZE,
        MAX_FACE_SIZE,
    )?;
    Ok(faces)
}

fn draw_box_around_face(frame: &mut Mat, face: Rect) -> Result<()> {
    println!("found face {:?}", face);
    let scaled_face = Rect {
        x: face.x * SCALE_FACTOR_INV,
        y: face.y * SCALE_FACTOR_INV,
        width: face.width * SCALE_FACTOR_INV,
        height: face.height * SCALE_FACTOR_INV,
    };

    const THICKNESS: i32 = 2;
    const LINE_TYPE: i32 = 8;
    const SHIFT: i32 = 0;
    let color_red = Scalar::new(0f64, 0f64, 255f64, -1f64);

    imgproc::rectangle(frame, scaled_face, color_red, THICKNESS, LINE_TYPE, SHIFT)?;
    Ok(())
}

fn blur_face(frame: &mut Mat, face: Rect) -> Result<()> {
    let frame_copy = frame.clone();
    let face_copy = Mat::roi(&frame_copy, face).unwrap();

    let mut blurred = Mat::default();
    imgproc::gaussian_blur(
        &face_copy,
        &mut blurred,
        Size::new(BLUR_STRENGTH, BLUR_STRENGTH),
        0f64,
        0f64,
        0,
    )?;

    let mut inset_image = Mat::roi(&frame, face).unwrap();

    blurred.copy_to(&mut inset_image)?;

    frame.copy_to(&mut blurred)?;
    Ok(())
}

fn frame_loop(
    mut capture: Capture,
    mut classifier: &mut objdetect::CascadeClassifier,
    window: Window,
) -> Result<()> {
    loop {
        let mut frame = match capture.grab_frame()? {
            Some(frame) => frame,
            None => continue,
        };

        let preprocessed = preprocess_image(&frame)?;
        let faces = detect_faces(&mut classifier, preprocessed)?;
        for face in faces {
            if USE_BLUR {
                blur_face(&mut frame, face)?;
            } else {
                draw_box_around_face(&mut frame, face)?;
            }
        }

        window.show_image(&frame)?;

        let key = highgui::wait_key(1)?;
        if key == Q_KEY_CODE {
            break;
        }
    }
    Ok(())
}
fn main() -> Result<()> {
    highgui::named_window("window", highgui::WINDOW_FULLSCREEN)?;

    let capture = Capture::create(0, CAPTURE_WIDTH, CAPTURE_HEIGHT)?;

    let mut classifier = objdetect::CascadeClassifier::new(CASCADE_XML_FILE)?;

    let window = Window::create("window", CAPTURE_WIDTH, CAPTURE_HEIGHT)?;

    if capture.is_opened()? {
        frame_loop(capture, &mut classifier, window)?;
    }

    Ok(())
}
