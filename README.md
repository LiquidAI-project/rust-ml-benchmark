# rust-ml-benchmark

This repository contains the code to benchmark ML inference using Rust.


## Folder Structure

- [src](src folder)
    - Contains the main.rs Rust program containing the benchmark as well as the inference code
- [assets/](assets/) folder
    - Contains the assets including models and images which will be used for ML inference

## Usage

Following is the cargo command format:

```bash
cargo run <model> <image>
```

### Parameters:
- <model> 
    - path of the model which will be used for inference

- <image>
    - path of the image which will be used for inference

Sample Command:
```bash
cargo run "assets/models/mobilenetv2-10.onnx" "assets/images/dog.jpg"
```
