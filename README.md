# rust-ml-benchmark

This repository contains the code to benchmark ML inference using Rust.


## Folder Structure

- [src/main.rs](src/main.rs) 
    - Contains the main.rs Rust program containing the benchmark as well as the inference code
- [assets/](assets/) folder
    - Contains the assets including models and images which will be used for ML inference
- [scripts/](scripts/) folder
    - Contains the scripts for automation. For example, to run Rust benchmark on a machine for a certain number of times

## Usage

Following is the cargo command format:

```bash
cargo run <model_path> <image_path>
```

Parameters:
- <model_path> 
    - path of the model which will be used for inference

- <image_path>
    - path of the image which will be used for inference

Sample Command:

Debug Mode:
```bash
cargo run "assets/models/mobilenetv2-10.onnx" "assets/imgs/unseen_dog.jpg"
```

Release Mode:
```bash
cargo run --release "assets/models/mobilenetv2-10.onnx" "assets/imgs/unseen_dog.jpg"
```

## Scripts:

- [scripts/benchmark.c](scripts/benchmark.c)
    - C program to run the benchmark for n number of times provided in the command line arguments

Usage:

```bash
cd scripts
gcc benchmark.c -o binary_file_name
```
Replace the binary_file_name with your desired name.

Command format:

```bash
./binary_file_name <num_iterations> <model_path> <image_path>
```

Parameters:
- <num_iterations>
    - number of times to run the Rust benchmark

- <model_path> 
    - path of the model which will be used for inference in the benchmarks

- <image_path>
    - path of the image which will be used for inference in the benchmarks

Sample Command:
```bash
gcc benchmark.c -o benchmark
./benchmark 10 "../assets/models/mobilenetv2-10.onnx" "../assets/imgs/unseen_dog.jpg"
```