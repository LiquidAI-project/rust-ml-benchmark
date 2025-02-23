use image::{imageops::FilterType, GenericImageView};
use ndarray::Array;
use ort::{execution_providers::CUDAExecutionProvider, session::Session};
use std::{env, time::Instant};
use sysinfo::{Pid, System};

fn main() -> ort::Result<()> {
    let mut system = System::new_all();
    system.refresh_all();

    let start_time = Instant::now();

    let args: Vec<String> = env::args().collect();
    if args.len() != 3 {
        println!("Usage: {} <model> <image>", args[0]);
        return Ok(());
    }

    let model: &str = &args[1];
    let image: &str = &args[2];

    ort::init()
        .with_execution_providers([CUDAExecutionProvider::default().build()])
        .commit()?;

    let original_img = image::open(image).unwrap();
    let (img_width, img_height) = (original_img.width(), original_img.height());

    println!(
        "Loaded image with height {:?} and width {:?} ",
        img_height, img_width
    );

    let img = original_img.resize_exact(224, 224, FilterType::CatmullRom);
    let mut input = Array::zeros((1, 3, 224, 224));
    for pixel in img.pixels() {
        let x = pixel.0 as _;
        let y = pixel.1 as _;
        let [r, g, b, _] = pixel.2 .0;
        input[[0, 0, y, x]] = (r as f32) / 255.;
        input[[0, 1, y, x]] = (g as f32) / 255.;
        input[[0, 2, y, x]] = (b as f32) / 255.;
    }

    let model = Session::builder()?.commit_from_file(model)?;

    let outputs = model.run(ort::inputs![input]?)?;

    let output_tensor = outputs[0].try_extract_tensor::<f32>()?;

    let output_array = output_tensor.view();
    let (predicted_index, &score) = output_array
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap();

    println!("Predicted Class Index: {}", predicted_index);
    println!("Confidence Score: {:.4}", score);

    let wall_clock_time = start_time.elapsed();

    let pid = std::process::id();
    let process = system
        .process(Pid::from_u32(pid))
        .expect("Process not found");

    let user_time = process.cpu_usage();
    let system_time = process.cpu_usage();
    let max_rss = process.memory();
    let cpu_usage = system.global_cpu_usage();

    println!("Wall clock time: {:?}", wall_clock_time);
    println!("CPU usage: {:.2}%", cpu_usage);
    println!("User time: {:.2}%", user_time);
    println!("System time: {:.2}%", system_time);
    println!("Max RSS: {} bytes", max_rss);

    Ok(())
}
