pub const CASCADE_XML_FILE: &str = "models/haarcascade_frontalface_alt.xml";

pub const USE_YUNET: bool = true;
pub const YUNET_MODEL_FILE: &str = "models/face_detection_yunet_2022mar.onnx";
pub const DNN_CONFIDENCE_THRESHOLD: f32 = 0.7;

pub const CAPTURE_WIDTH: i32 = 640;
pub const CAPTURE_HEIGHT: i32 = 480;

pub const SCALE_FACTOR: f64 = 1f64;
pub const SCALE_FACTOR_INV: i32 = (1f64 / SCALE_FACTOR) as i32;

pub const USE_BLUR: bool = true;
pub const BLUR_STRENGTH: i32 = 177;

pub const Q_KEY_CODE: i32 = 113;
