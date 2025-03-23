use image::{imageops::FilterType, DynamicImage, GenericImageView};
use ndarray::{Array, ArrayBase, Dim, IxDynImpl, OwnedRepr, ViewRepr};
use ort::{
    execution_providers::CUDAExecutionProvider,
    session::{Session, SessionOutputs},
};
use std::{
    env,
    time::{Duration, Instant},
};
use sysinfo::{Pid, Process, System};

struct LoadTimes {
    env_load_time: Duration,
    image_load_time: Duration,
    image_processing_time: Duration,
    model_load_time: Duration,
    model_run_time: Duration,
}

struct Metrics {
    wall_clock_time: Duration,
    user_time: f32,
    system_time: f32,
    max_rss: u64,
    swap_memory: u64,
}

fn main() -> ort::Result<()> {
    let mut system = System::new_all();
    system.refresh_all();

    let args: Vec<String> = env::args().collect();
    if args.len() != 3 {
        println!("Usage: {} <model> <image>", args[0]);
        return Ok(());
    }

    let model: &str = &args[1];
    let image: &str = &args[2];

    let start_time: Instant = Instant::now();

    ort::init()
        .with_execution_providers([CUDAExecutionProvider::default().build()])
        .commit()?;

    let env_load_time: Duration = start_time.elapsed();

    let original_img: DynamicImage = image::open(image).unwrap();
    let (img_width, img_height) = (original_img.width(), original_img.height());

    println!(
        "Loaded image with height {:?} and width {:?} ",
        img_height, img_width
    );
    let image_load_time: Duration = start_time.elapsed() - env_load_time;

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

    let image_processing_time: Duration = start_time.elapsed() - image_load_time;

    let model: Session = Session::builder()?.commit_from_file(model)?;

    let model_load_time: Duration = start_time.elapsed() - image_processing_time;

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
    let model_run_time: Duration = wall_clock_time - model_load_time;
    println!("Predicted Class Index: {}", predicted_index);
    println!("Confidence Score: {:.4}", score);

    let pid: u32 = std::process::id();
    let process: &Process = system
        .process(Pid::from_u32(pid))
        .expect("Process not found");

    let load_times = LoadTimes {
        env_load_time,
        image_load_time,
        image_processing_time,
        model_load_time,
        model_run_time,
    };

    let metrics = Metrics {
        wall_clock_time,
        user_time: process.cpu_usage(),
        system_time: process.cpu_usage(),
        max_rss: process.memory(),
        swap_memory: (process.virtual_memory() - process.memory()),
    };

    print_load_times(&load_times);
    print_metrics(&metrics);

    Ok(())
}

fn print_load_times(load_times: &LoadTimes) {
    println!();
    println!("=================Load times Results===================");
    println!("Env load time: {:?}", load_times.env_load_time);
    println!("Image load time: {:?}", load_times.image_load_time);
    println!("Image Processing time: {:?}", load_times.image_processing_time);
    println!("Model Load time: {:?}", load_times.model_load_time);
    println!("Model run took time: {:?}", load_times.model_run_time);
    println!("=======================================================");
    println!();
}

fn print_metrics(metrics: &Metrics) {
    println!("=================Benchmarking Results==================");
    println!("Wall Clock Time: {:?}", metrics.wall_clock_time);
    println!("CPU usage: {:.2}%", metrics.system_time);
    println!("User time: {:.2}%", metrics.user_time);
    println!("System time: {:.2}%", metrics.system_time);
    println!("Max RSS: {} bytes", metrics.max_rss);
    println!("Swap memory: {} bytes", metrics.swap_memory);
    println!("========================================================");
}