#include <stdlib.h>

int main() {
    for (int i = 0; i < 100; i++) {
        system("../target/release/rust-ml-benchmark \"../assets/models/mobilenetv2-10.onnx\" \"../assets/imgs/unseen_dog.jpg\"");
    }
}