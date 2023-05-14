mod capture;
mod constants;
mod window;

use crate::constants::*;
use anyhow::Result;
use capture::Capture;
use opencv::core::{Mat, Ptr, Rect, Scalar, Size, Vector, CV_32F};
use opencv::{
    dnn, highgui, imgproc, objdetect,
    prelude::*,
    types::{VectorOfMat, VectorOfRect},
};
use window::Window;

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

fn clamp_rect_to_image_bounds(rect: Rect) -> Rect {
    let mut rect = rect.clone();
    if rect.x < 0 {
        rect.x = 0;
    }
    if rect.y < 0 {
        rect.y = 0;
    }
    if rect.x + rect.width > CAPTURE_WIDTH {
        rect.width = CAPTURE_WIDTH - rect.x;
    }
    if rect.y + rect.height > CAPTURE_HEIGHT {
        rect.height = CAPTURE_HEIGHT - rect.y;
    }
    rect
}

fn detect_faces(
    classifiers: &mut Vec<&mut objdetect::CascadeClassifier>,
    image: Mat,
) -> Result<VectorOfRect> {
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

    let mut faces = VectorOfRect::new();

    for classifier in classifiers.iter_mut() {
        classifier.detect_multi_scale(
            &image,
            &mut faces,
            SCALE_FACTOR,
            MIN_NEIGHBORS,
            FLAGS,
            MIN_FACE_SIZE,
            MAX_FACE_SIZE,
        )?;
    }
    Ok(faces)
}

fn detect_faces_dnn(net: &mut dnn::Net, out_names: &Vector<String>, frame: &Mat) -> Result<()> {
    let frame_width = frame.size()?.width;
    let frame_height = frame.size()?.height;

    let mut blob = dnn::blob_from_image(
        &frame,
        1.0,
        Size::new(300, 300),
        Scalar::new(104f64, 177f64, 123f64, 0f64),
        false,
        false,
        CV_32F,
    )?;

    println!("blob {:?}", blob);

    net.set_input(&mut blob, "", 1.0, Scalar::default())?;

    let mut detections = VectorOfMat::new();

    net.forward(&mut detections, &out_names)?;

    let detection_result = detections.get(0)?;
    let num_detections = detection_result.rows();

    println!("hi");
    println!("num_detections: {:?}", num_detections);

    for detection in detections {
        let confidence = detection.at::<f32>(2)?;
        let x_left_bottom = *(detection.at::<f32>(3)?) as i32 * frame_width;
        let y_left_bottom = *(detection.at::<f32>(4)?) as i32 * frame_height;
        let x_right_top = *(detection.at::<f32>(5)?) as i32 * frame_width;
        let y_right_top = *(detection.at::<f32>(6)?) as i32 * frame_height;

        if confidence > &DNN_CONFIDENCE_THRESHOLD {
            let face = Rect {
                x: x_left_bottom,
                y: y_left_bottom,
                width: x_right_top - x_left_bottom,
                height: y_right_top - y_left_bottom,
            };

            println!("Face detected: {:?}", face);
        }
    }

    Ok(())
}

fn detect_faces_yunet(
    face_detector: &mut Ptr<dyn FaceDetectorYN>,
    frame: &Mat,
) -> Result<VectorOfRect> {
    let mut detections = Mat::default();
    let mut faces = VectorOfRect::new();

    face_detector.detect(frame, &mut detections)?;

    for i in 0..detections.rows() {
        let confidence = detections.at_2d::<f32>(i, 14)?;
        let x1 = *(detections.at_2d::<f32>(i, 0)?) as i32;
        let y1 = *(detections.at_2d::<f32>(i, 1)?) as i32;
        let w = *(detections.at_2d::<f32>(i, 2)?) as i32;
        let h = *(detections.at_2d::<f32>(i, 3)?) as i32;

        if confidence > &DNN_CONFIDENCE_THRESHOLD {
            let face = Rect {
                x: x1,
                y: y1,
                width: w,
                height: h,
            };

            faces.push(face);

            println!("Face detected: {:?}", face);
        }
    }

    Ok(faces)
}

fn draw_box_around_face(frame: &mut Mat, face: Rect) -> Result<()> {
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
    let clamped_face = clamp_rect_to_image_bounds(face);
    let face_roi = Mat::roi(&frame_copy, clamped_face).unwrap();

    let mut blurred = Mat::default();
    imgproc::gaussian_blur(
        &face_roi,
        &mut blurred,
        Size::new(BLUR_STRENGTH, BLUR_STRENGTH),
        0f64,
        0f64,
        0,
    )?;

    let mut inset_image = Mat::roi(&frame, clamped_face).unwrap();

    blurred.copy_to(&mut inset_image)?;

    frame.copy_to(&mut blurred)?;
    Ok(())
}

fn frame_loop(
    mut capture: Capture,
    classifiers: &mut Vec<&mut objdetect::CascadeClassifier>,
    //net: &mut dnn::Net,
    face_detector: &mut Ptr<dyn FaceDetectorYN>,
    //out_names: &Vector<String>,
    window: Window,
) -> Result<()> {
    loop {
        let mut frame = match capture.grab_frame()? {
            Some(frame) => frame,
            None => continue,
        };

        /* let preprocessed = preprocess_image(&frame)?;
        let faces = detect_faces(classifiers, preprocessed)?;
        for face in faces {
            println!("found face {:?}", face);
            if USE_BLUR {
                blur_face(&mut frame, face)?;
            } else {
                draw_box_around_face(&mut frame, face)?;
            }
        } */
        //detect_faces_dnn(net, out_names, &frame)?;

        let faces = detect_faces_yunet(face_detector, &frame)?;
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
    let mut classifiers = vec![&mut classifier];

    /* let mut net = dnn::read_net(DNN_CAFFE_MODEL_FILE, DNN_CAFFE_CONFIG_FILE, "Caffe")?;
    net.set_preferable_backend(dnn::DNN_BACKEND_CUDA)?;
    net.set_preferable_target(dnn::DNN_TARGET_CUDA)?;
    let out_names = net.get_unconnected_out_layers_names()?; */

    let mut face_detector: Ptr<dyn FaceDetectorYN> = <dyn objdetect::FaceDetectorYN>::create(
        "models/face_detection_yunet_2022mar.onnx",
        "",
        Size::new(CAPTURE_WIDTH, CAPTURE_HEIGHT),
        0.9f32,
        0.3f32,
        5000,
        dnn::DNN_BACKEND_CUDA,
        dnn::DNN_TARGET_CUDA,
    )?;

    let window = Window::create("window", CAPTURE_WIDTH, CAPTURE_HEIGHT)?;

    if capture.is_opened()? {
        frame_loop(
            capture,
            &mut classifiers,
            &mut face_detector,
            //&out_names,
            window,
        )?;
    }

    Ok(())
}
