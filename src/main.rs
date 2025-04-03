use image::{imageops::FilterType, DynamicImage, GenericImageView};
use libc::{getrusage, rusage, RUSAGE_SELF};
use ndarray::{Array, ArrayBase, Dim, IxDynImpl, OwnedRepr, ViewRepr};
use ort::{
    execution_providers::CUDAExecutionProvider,
    session::{Session, SessionOutputs},
    Error as OrtError,
};
use std::{
    env,
    time::{Duration, Instant},
};
use thiserror::Error;

#[derive(Debug, Error)]
enum AppError {
    #[error("Usage: {} <model> <image>", .0)]
    UsageError(String),
    #[error("Failed to load image: {0}")]
    ImageLoadError(#[from] image::ImageError),
    #[error("ORT error: {0}")]
    OrtError(#[from] OrtError),
}

#[derive(Debug, Clone)]
struct Metrics {
    name: String,
    timestamp: Instant,
    wall_clock_time: Duration,
    user_time: Duration,
    system_time: Duration,
    max_rss: u64,
}

impl Metrics {
    fn current(name: String) -> Self {
        unsafe {
            let mut usage: rusage = std::mem::zeroed();
            getrusage(RUSAGE_SELF, &mut usage);

            let user_time: Duration = Duration::from_secs(usage.ru_utime.tv_sec as u64)
                + Duration::from_micros(usage.ru_utime.tv_usec as u64);

            let system_time: Duration = Duration::from_secs(usage.ru_stime.tv_sec as u64)
                + Duration::from_micros(usage.ru_stime.tv_usec as u64);

            Self {
                name,
                timestamp: Instant::now(),
                wall_clock_time: Duration::default(),
                user_time,
                system_time,
                max_rss: usage.ru_maxrss as u64,
            }
        }
    }

    fn diff(&self, prev: &Self) -> Self {
        Self {
            name: self.name.clone(),
            timestamp: self.timestamp,
            wall_clock_time: self.timestamp.duration_since(prev.timestamp),
            user_time: self.user_time - prev.user_time,
            system_time: self.system_time - prev.system_time,
            max_rss: self.max_rss - prev.max_rss,
        }
    }
}

impl std::fmt::Display for Metrics {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "============= {} Metrics =============", self.name)?;
        writeln!(f, "Wall Clock Time: {:?}", self.wall_clock_time)?;
        writeln!(f, "User time: {:?}", self.user_time)?;
        writeln!(f, "System time: {:?}", self.system_time)?;
        writeln!(f, "Max RSS: {} bytes", self.max_rss)?;
        writeln!(f, "=======================================")
    }
}

#[derive(Debug)]
struct BenchmarkTracker {
    start_metrics: Metrics,
    current_operation: Option<Metrics>,
    completed_metrics: Vec<Metrics>,
}

impl BenchmarkTracker {
    fn new() -> Self {
        Self {
            start_metrics: Metrics::current("Total".to_string()),
            current_operation: None,
            completed_metrics: Vec::new(),
        }
    }

    fn start_operation(&mut self, name: &str) {
        self.current_operation = Some(Metrics::current(name.to_string()));
    }

    fn finish_operation(&mut self) {
        if let Some(start_metrics) = self.current_operation.take() {
            self.finish_operation_internal(start_metrics);
        }
    }

    fn finish_operation_internal(&mut self, start_metrics: Metrics) {
        let end_metrics: Metrics = Metrics::current(start_metrics.name.clone());
        let diff_metrics: Metrics = end_metrics.diff(&start_metrics);

        self.completed_metrics.push(diff_metrics);
    }

    fn get_total_metrics(&self) -> Metrics {
        let current: Metrics = Metrics::current("Total".to_string());
        current.diff(&self.start_metrics)
    }

    fn print_all_metrics(&self) {
        let total: Metrics = self.get_total_metrics();
        
        for metrics in &self.completed_metrics {
            print!("{}", metrics);
        }

        print!("{}", total);
    }
}

fn load_model(model_path: &str) -> Result<Session, OrtError> {
    let model: Session = Session::builder()?.commit_from_file(model_path)?;
    Ok(model)
}

fn process_image(original_img: DynamicImage) -> ArrayBase<OwnedRepr<f32>, Dim<[usize; 4]>> {
    let img: DynamicImage = original_img.resize_exact(224, 224, FilterType::CatmullRom);
    let mut input: ArrayBase<OwnedRepr<f32>, Dim<[usize; 4]>> = Array::zeros((1, 3, 224, 224));
    for pixel in img.pixels() {
        let x: usize = pixel.0 as _;
        let y: usize = pixel.1 as _;
        let [r, g, b, _] = pixel.2 .0;
        input[[0, 0, y, x]] = (r as f32) / 255.;
        input[[0, 1, y, x]] = (g as f32) / 255.;
        input[[0, 2, y, x]] = (b as f32) / 255.;
    }
    input
}

fn main() -> Result<(), AppError> {
    let args: Vec<String> = env::args().collect();
    if args.len() != 3 {
        return Err(AppError::UsageError(args[0].clone()));
    }

    let model_path: &str = &args[1];
    let image_path: &str = &args[2];

    let mut tracker: BenchmarkTracker = BenchmarkTracker::new();

    tracker.start_operation("Env Load");
    ort::init()
        .with_execution_providers([CUDAExecutionProvider::default().build()])
        .commit()?;
    tracker.finish_operation();

    tracker.start_operation("Image Load");
    let original_img: DynamicImage = image::open(image_path)?;
    tracker.finish_operation();

    tracker.start_operation("Image Processing");
    let input: ArrayBase<OwnedRepr<f32>, Dim<[usize; 4]>> = process_image(original_img);
    tracker.finish_operation();

    tracker.start_operation("Model Load");
    let model: Session = load_model(model_path).map_err(AppError::OrtError)?;
    tracker.finish_operation();

    tracker.start_operation("Model Run");
    let outputs: SessionOutputs<'_, '_> = model.run(ort::inputs![input]?)?;

    let output_tensor: ArrayBase<ViewRepr<&f32>, Dim<IxDynImpl>> =
        outputs[0].try_extract_tensor::<f32>()?;

    let output_array: ArrayBase<ViewRepr<&f32>, Dim<IxDynImpl>> = output_tensor.view();
    let (predicted_index, &score) = output_array
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap();
    tracker.finish_operation();

    tracker.print_all_metrics();

    println!("Predicted Class Index: {}", predicted_index);
    println!("Confidence Score: {:.4}", score);

    Ok(())
}
