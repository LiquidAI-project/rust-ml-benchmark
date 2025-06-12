# rust-ml-benchmark

This branch automates the deployment and benchmarking of a Rust-based machine learning inference system using Ansible. It supports running ONNX model inference remotely (e.g., on a Raspberry Pi), collecting performance metrics, and storing results locally for further analysis.

---

## Project Structure

```
rust-ml-benchmark/
├── Ansible/                  # Automation with Ansible
│   ├── Benchmark.yaml        # Ansible playbook for remote deployment and execution
│   ├── ansible.cfg           # Ansible configuration
│   └── inventory/
│       └── hosts.ini         # Inventory of remote target devices
├── Benchmark-Setup/         # Benchmark source code and resources
│   ├── Cargo.toml
│   ├── src/
│   │   └── main.rs 
│   ├── assets/
│   │   ├── imgs/
│   │   └── models/
│   ├── scripts/              # Benchmark and logging scripts + results
│   └── README.md 
└── Results/                  # Benchmark results fetched from remote devices
```

---

## Automation with Ansible

Ansible is used to automate the entire workflow:

1. **Install** rust if not installed on the remote device
2. **Transfer source code and assets** to the remote device
3. **Run the benchmark script** present in the scripts folder in the Benchmark setup
3. **Collect result files** and copy them back into the `Results/` directory

### Configuration

- `Ansible/Benchmark.yaml` is the main playbook orchestrating the process.
- `Ansible/inventory/hosts.ini` defines the target host(s).
- SSH keys or passwords must be configured for Ansible to access the devices.

### Run the Deployment

```bash
cd Ansible
ansible-playbook -i inventory/hosts.ini Benchmark.yaml
```

This will:
- Copy the contents of `Benchmark-Setup` to the remote machine
- Compile and run the benchmark
- Collect all resulting `.csv` files from the remote into the local `Results/` directory

---

## Benchmark Details

The benchmarking process measures the following stages in the ML inference pipeline:

- Image Reading (`readimg.csv`)
- Model Loading (`loadmodel.csv`)
- Inference (`inference.csv`)
- Postprocessing (`postprocessing.csv`)
- Time taken by the loading processes (`redbox.csv`)
- Time taken by the inference and other processes (`greenbox.csv`)
- Total Time (`total.csv`)

Each result is collected in `Results/`

---

## Dependencies

Ensure the following are installed before running the automation:
- ansible on your device
- gcc is installed on the remote device

---

## License

This project is licensed under the MIT License. See the `Benchmark-Setup/LICENSE` file for details.

---
