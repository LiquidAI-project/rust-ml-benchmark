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

#[derive(Debug)]
struct LoadTimes {
    env_load_time: Duration,
    image_load_time: Duration,
    image_processing_time: Duration,
    model_load_time: Duration,
    model_run_time: Duration,
}

#[derive(Debug)]
struct Metrics {
    wall_clock_time: Duration,
    user_time: Duration,
    system_time: Duration,
    max_rss: u64,
}

fn get_benchmark_metrics() -> (Duration, Duration, u64) {
    unsafe {
        let mut usage: rusage = std::mem::zeroed();
        getrusage(RUSAGE_SELF, &mut usage);

        let user_time: Duration = Duration::from_secs(usage.ru_utime.tv_sec as u64)
            + Duration::from_micros(usage.ru_utime.tv_usec as u64);

        let system_time: Duration = Duration::from_secs(usage.ru_stime.tv_sec as u64)
            + Duration::from_micros(usage.ru_stime.tv_usec as u64);
        
        let max_rss : u64 = usage.ru_maxrss as u64;

        (user_time, system_time, max_rss)
    }
}

fn load_model(model_path: &str) -> Result<Session, OrtError> {
    let model = Session::builder()?.commit_from_file(model_path)?;
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

    let start_time: Instant = Instant::now();

    ort::init()
        .with_execution_providers([CUDAExecutionProvider::default().build()])
        .commit()?;

    let env_load_time: Duration = start_time.elapsed();

    let original_img: DynamicImage = image::open(image_path)?;
    let (img_width, img_height) = (original_img.width(), original_img.height());

    println!(
        "Loaded image with height {:?} and width {:?} ",
        img_height, img_width
    );
    let image_load_time: Duration = start_time.elapsed();

    let input: ArrayBase<OwnedRepr<f32>, Dim<[usize; 4]>> = process_image(original_img);

    let image_processing_time: Duration = start_time.elapsed();

    let model = load_model(model_path).map_err(AppError::OrtError)?;

    let model_load_time: Duration = start_time.elapsed();

    let outputs: SessionOutputs<'_, '_> = model.run(ort::inputs![input]?)?;

    let output_tensor: ArrayBase<ViewRepr<&f32>, Dim<IxDynImpl>> =
        outputs[0].try_extract_tensor::<f32>()?;

    let output_array: ArrayBase<ViewRepr<&f32>, Dim<IxDynImpl>> = output_tensor.view();
    let (predicted_index, &score) = output_array
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap();

    let wall_clock_time: Duration = start_time.elapsed();
    let model_run_time: Duration = start_time.elapsed();
    println!("Predicted Class Index: {}", predicted_index);
    println!("Confidence Score: {:.4}", score);

    let (user_time, system_time, max_rss) = get_benchmark_metrics();

    let load_times = LoadTimes {
        env_load_time,
        image_load_time,
        image_processing_time,
        model_load_time,
        model_run_time,
    };

    let metrics: Metrics = Metrics {
        wall_clock_time,
        user_time,
        system_time,
        max_rss,
    };

    print_load_times(&load_times);
    print_metrics(&metrics);

    Ok(())
}

fn print_load_times(load_times: &LoadTimes) {
    println!();
    println!("=================Load times Results===================");
    println!("Env load time: {:?}", load_times.env_load_time);
    println!(
        "Image load time: {:?}",
        (load_times.image_load_time - load_times.env_load_time)
    );
    println!(
        "Image Processing time: {:?}",
        (load_times.image_processing_time - load_times.image_load_time)
    );
    println!(
        "Model Load time: {:?}",
        (load_times.model_load_time - load_times.image_processing_time)
    );
    println!(
        "Model run took time: {:?}",
        (load_times.model_run_time - load_times.model_load_time)
    );
    println!("=======================================================");
    println!();
}

fn print_metrics(metrics: &Metrics) {
    println!("=================Benchmarking Results==================");
    println!("Wall Clock Time: {:?}", metrics.wall_clock_time);
    println!("User time: {:?}", metrics.user_time);
    println!("System time: {:?}", metrics.system_time);
    println!("Max RSS: {} bytes", metrics.max_rss);
    println!("========================================================");
}
