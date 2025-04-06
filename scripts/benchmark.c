#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <sys/stat.h>
#include <sys/types.h>

#define MAX_LINE_LEN 512

typedef struct
{
    float user_time;
    float system_time;
    float cpu_usage;
    float wall_clock;
    long max_rss;
} Metrics;

int parse_line(char *line, const char *prefix, float *val, const char *unit)
{
    char *p = strstr(line, prefix);
    if (p)
    {
        p += strlen(prefix);
        if (unit && strstr(p, unit))
        {
            *strstr(p, unit) = '\0';
        }
        *val = atof(p);
        return 1;
    }
    return 0;
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

int parse_metrics_block(FILE *fp, Metrics *out)
{
    char line[MAX_LINE_LEN];
    memset(out, 0, sizeof(Metrics));
    int found = 0;

    while (fgets(line, sizeof(line), fp))
    {
        if (strstr(line, "Wall Clock Time:"))
        {
            parse_line(line, "Wall Clock Time:", &out->wall_clock, "ms");
            found++;
        }
        else if (strstr(line, "User time:"))
        {
            parse_line(line, "User time:", &out->user_time, "ms");
            found++;
        }
        else if (strstr(line, "System time:"))
        {
            parse_line(line, "System time:", &out->system_time, "ms");
            found++;
        }
        else if (strstr(line, "CPU Usage:"))
        {
            parse_line(line, "CPU Usage:", &out->cpu_usage, "%");
            found++;
        }
        else if (strstr(line, "Max RSS:"))
        {
            parse_rss(line, &out->max_rss);
            found++;
        }
        else if (strstr(line, "======================================="))
        {
            break;
        }
    }

    return found == 5;
}

int write_csv_header(FILE *file, const char *data)
{
    return fprintf(file, "%s\n", data);
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

int main(int argc, char *argv[])
{
    int num_iterations;
    const char *model_path;
    const char *image_path;

    if (check_args(argc, argv, &num_iterations, &model_path, &image_path))
    {
        return 1;
    }

    if (mkdir("bench", 0777) == -1 && errno != EEXIST)
    {
        perror("Failed to create directory");
        return 1;
    }

    FILE *total_metrics_csv = fopen("./bench/total.csv", "w");
    if (!total_metrics_csv)
    {
        fprintf(stderr, "Failed to open CSV file\n");
        return 1;
    }
    write_csv_header(total_metrics_csv, "user_time,system_time,cpu_percent,wallclock_time,max_rss");

    for (int i = 1; i <= num_iterations; i++)
    {
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
            if (strstr(line, "Total Metrics"))
            {
                if (parse_metrics_block(fp, &m))
                {
                    write_csv(total_metrics_csv, &m);
                }
            }
        }

        fclose(fp);
    }

    fclose(total_metrics_csv);

    return 0;
}