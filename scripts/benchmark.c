#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[])
{
    if (argc != 4)
    {
        fprintf(stderr, "Usage: %s <num_iterations> <model_path> <image_path>\n", argv[0]);
        return 1;
    }

    const num_iterations = atoi(argv[1]);

    if (num_iterations <= 0)
    {
        fprintf(stderr, "Error: Number of iterations must be a positive integer\n");
        return 1;
    }

    const char *model_path = argv[2];
    const char *image_path = argv[3];

    if (strlen(model_path) == 0 || strlen(image_path) == 0)
    {
        fprintf(stderr, "Error: Model path or image path could not be empty \n");
        return 1;
    }

    for (int i = 0; i < 100; i++)
    {
        char command[1024];
        snprintf(command, sizeof(command),
                 "../target/release/rust-ml-benchmark \"%s\" \"%s\"",
                 model_path, image_path);
        system(command);
    }
}