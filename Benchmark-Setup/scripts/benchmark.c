#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <sys/stat.h>
#include <sys/types.h>

#define MAX_LINE_LEN 512

typedef struct
{
    char *name;
    float user_time;
    float system_time;
    float cpu_usage;
    float wall_clock;
    long max_rss;
} Metrics;

int parse_time_line(char *line, const char *prefix, float *val)
{
    char *p = strstr(line, prefix);
    if (!p)
        return 0;

    p += strlen(prefix);
    while (*p == ' ')
        p++;

    float number = 0.0f;
    char unit[10] = {0};

    if (sscanf(p, "%f%9s", &number, unit) < 1)
        return 0;

    if (strcmp(unit, "s") == 0 || strcmp(unit, "sec") == 0)
        number *= 1000;
    else if (strcmp(unit, "µs") == 0 || strcmp(unit, "microseconds") == 0)
        number /= 1000;

    *val = number;
    return 1;
}

int parse_cpu_line(char *line, const char *prefix, float *val)
{
    char *p = strstr(line, prefix);
    if (!p)
        return 0;

    p += strlen(prefix);
    while (*p == ' ')
        p++;

    float number = 0.0f;
    sscanf(p, "%f", &number);

    *val = number;
    return 1;
}

int parse_rss(char *line, long *rss)
{
    char *p = strstr(line, "Max RSS:");
    if (p)
    {
        p += strlen("Max RSS:");
        *rss = atol(p);
        return 1;
    }
    return 0;
}

float calculate_new_average(float old_avg, int currentCount, float current_value)
{
    if (currentCount == 0)
        return current_value;
    return (currentCount * old_avg + current_value) / (currentCount + 1);
}

int parse_metrics_block(FILE *fp, Metrics *out, Metrics *avg_metrics, int currentCount)
{
    char line[MAX_LINE_LEN];
    memset(out, 0, sizeof(Metrics));
    int found = 0;

    while (fgets(line, sizeof(line), fp))
    {
        if (strstr(line, "Wall Clock Time:"))
        {
            parse_time_line(line, "Wall Clock Time:", &out->wall_clock);
            avg_metrics->wall_clock = calculate_new_average(avg_metrics->wall_clock, currentCount - 1, out->wall_clock);
            found++;
        }
        else if (strstr(line, "User time:"))
        {
            parse_time_line(line, "User time:", &out->user_time);
            avg_metrics->user_time = calculate_new_average(avg_metrics->user_time, currentCount - 1, out->user_time);
            found++;
        }
        else if (strstr(line, "System time:"))
        {
            parse_time_line(line, "System time:", &out->system_time);
            avg_metrics->system_time = calculate_new_average(avg_metrics->system_time, currentCount - 1, out->system_time);
            found++;
        }
        else if (strstr(line, "CPU Usage:"))
        {
            parse_cpu_line(line, "CPU Usage:", &out->cpu_usage);
            avg_metrics->cpu_usage = calculate_new_average(avg_metrics->cpu_usage, currentCount - 1, out->cpu_usage);
            found++;
        }
        else if (strstr(line, "Max RSS:"))
        {
            parse_rss(line, &out->max_rss);
            avg_metrics->max_rss = (((currentCount - 1) * avg_metrics->max_rss) + out->max_rss) / (currentCount);
            found++;
        }
        else if (strstr(line, "======================================="))
        {
            break;
        }
    }

    return found == 5;
}

int write_csv_header(FILE *file)
{
    return fprintf(file, "%s\n", "user_time,system_time,cpu_percent,wallclock_time,max_rss");
}

void write_csv(FILE *file, Metrics *m)
{
    fprintf(file, "%.3f,%.3f,%.2f%%,%.3f,%ld\n", m->user_time, m->system_time, m->cpu_usage, m->wall_clock, m->max_rss);
}

int check_args(int argc, char *argv[], int *num_iterations, const char **model_path, const char **image_path)
{
    if (argc != 4)
    {
        fprintf(stderr, "Usage: %s <num_iterations> <model_path> <image_path>\n", argv[0]);
        return 1;
    }

    *num_iterations = atoi(argv[1]);
    if (*num_iterations <= 0)
    {
        fprintf(stderr, "Error: Number of iterations must be a positive integer\n");
        return 1;
    }

    *model_path = argv[2];
    *image_path = argv[3];

    if (strlen(*model_path) == 0 || strlen(*image_path) == 0)
    {
        fprintf(stderr, "Error: Model path or image path could not be empty\n");
        return 1;
    }

    return 0;
}

void print_metrics(Metrics *m, char *name)
{
    printf("====%s Metrics====\n", name);
    printf("Average Wall Clock Time: %.3f ms\n", m->wall_clock);
    printf("Average User Time: %.3f ms\n", m->user_time);
    printf("Average System Time: %.3f ms\n", m->system_time);
    printf("Average Cpu Usage: %.2f %%\n", m->cpu_usage);
    printf("Average Max RSS: %ld\n", m->max_rss);
}

int main(int argc, char *argv[])
{
    int num_iterations;
    const char *model_path;
    const char *image_path;
    Metrics avg_loadmodel_metrics = {0};
    Metrics avg_readimg_metrics = {0};
    Metrics avg_redbox_metrics = {0};
    Metrics avg_readimg_green_metrics = {0};
    Metrics avg_inference_metrics = {0};
    Metrics avg_postprocessing_metrics = {0};
    Metrics avg_greenbox_metrics = {0};
    Metrics avg_total_metrics = {0};

    int current_count = 0;

    if (check_args(argc, argv, &num_iterations, &model_path, &image_path))
    {
        return 1;
    }

    if (mkdir("bench", 0777) == -1 && errno != EEXIST)
    {
        perror("Failed to create directory");
        return 1;
    }

    FILE *loadmodel_metrics_csv = fopen("./bench/loadmodel.csv", "w");
    FILE *readimg_metrics_csv = fopen("./bench/readimg.csv", "w");
    FILE *redbox_metrics_csv = fopen("./bench/redbox.csv", "w");
    FILE *readimg_greenbox_csv = fopen("./bench/readimg_greenbox.csv", "w");
    FILE *inference_csv = fopen("./bench/inference.csv", "w");
    FILE *postprocessing_csv = fopen("./bench/postprocessing.csv", "w");
    FILE *greenbox_metrics_csv = fopen("./bench/greenbox.csv", "w");
    FILE *total_metrics_csv = fopen("./bench/total.csv", "w");
    if (!loadmodel_metrics_csv || !readimg_metrics_csv || !redbox_metrics_csv || !readimg_greenbox_csv || !inference_csv || !postprocessing_csv || !greenbox_metrics_csv || !total_metrics_csv)
    {
        fprintf(stderr, "Failed to open CSV file\n");
        return 1;
    }

    write_csv_header(loadmodel_metrics_csv);
    write_csv_header(readimg_metrics_csv);
    write_csv_header(redbox_metrics_csv);
    write_csv_header(readimg_greenbox_csv);
    write_csv_header(inference_csv);
    write_csv_header(postprocessing_csv);
    write_csv_header(greenbox_metrics_csv);
    write_csv_header(total_metrics_csv);

    char build_command[1024];
    snprintf(build_command, sizeof(build_command),
             "MODEL_PATH=\"%s\" IMAGE_PATH=\"%s\" cargo build --release --manifest-path=../Cargo.toml",
             model_path, image_path);

    if (system(build_command) != 0)
    {
        fprintf(stderr, "Rust compilation failed.\n");
        return 1;
    }

    for (int i = 1; i <= num_iterations; i++)
    {
        current_count++;
        printf("Running iteration %d\n", i);
        char command[1024];
        snprintf(command, sizeof(command),
                 "../target/release/rust-ml-benchmark \"%s\" \"%s\" > tmp_output.txt",
                 model_path, image_path);

        if (system(command) != 0)
        {
            fprintf(stderr, "Command failed on iteration %d", i);
            continue;
        }

        FILE *fp = fopen("tmp_output.txt", "r");
        if (!fp)
        {
            continue;
        }

        char line[MAX_LINE_LEN];
        while (fgets(line, sizeof(line), fp))
        {
            Metrics m;
            if (strstr(line, "loadmodel Metrics"))
            {
                if (parse_metrics_block(fp, &m, &avg_loadmodel_metrics, current_count))
                    write_csv(loadmodel_metrics_csv, &m);
            }
            else if (strstr(line, "readimg Metrics"))
            {
                if (parse_metrics_block(fp, &m, &avg_readimg_metrics, current_count))
                    write_csv(readimg_metrics_csv, &m);
            }
            else if (strstr(line, "RED BOX Phase Metrics"))
            {
                if (parse_metrics_block(fp, &m, &avg_redbox_metrics, current_count))
                    write_csv(redbox_metrics_csv, &m);
            }
            else if (strstr(line, "Pre-processing Metrics"))
            {
                if (parse_metrics_block(fp, &m, &avg_readimg_green_metrics, current_count))
                    write_csv(readimg_greenbox_csv, &m);
            }
            else if (strstr(line, "Inference Metrics"))
            {
                if (parse_metrics_block(fp, &m, &avg_inference_metrics, current_count))
                    write_csv(inference_csv, &m);
            }
            else if (strstr(line, "Post-processing Metrics"))
            {
                if (parse_metrics_block(fp, &m, &avg_postprocessing_metrics, current_count))
                    write_csv(postprocessing_csv, &m);
            }
            else if (strstr(line, "GREEN BOX Phase Metrics"))
            {
                if (parse_metrics_block(fp, &m, &avg_greenbox_metrics, current_count))
                    write_csv(greenbox_metrics_csv, &m);
            }
            else if (strstr(line, "Total Metrics"))
            {
                if (parse_metrics_block(fp, &m, &avg_total_metrics, current_count))
                    write_csv(total_metrics_csv, &m);
            }
        }

        fclose(fp);
    }

    fclose(loadmodel_metrics_csv);
    fclose(readimg_metrics_csv);
    fclose(redbox_metrics_csv);
    fclose(readimg_greenbox_csv);
    fclose(inference_csv);
    fclose(postprocessing_csv);
    fclose(greenbox_metrics_csv);
    fclose(total_metrics_csv);

    printf("Benchmarking completed. CSV files generated \n");

    print_metrics(&avg_loadmodel_metrics, "Load Model");
    print_metrics(&avg_readimg_metrics, "Read Image");
    print_metrics(&avg_readimg_green_metrics, "Pre Processing");
    print_metrics(&avg_inference_metrics, "Inference");
    print_metrics(&avg_postprocessing_metrics, "Post Processing");

    return 0;
}