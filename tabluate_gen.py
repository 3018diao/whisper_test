from tabulate import tabulate

def create_comparison_table(data):
    headers = ["Model", "Params (B)", "CPU Time (s)", "GPU Time (s)", "GPU Speedup"]
    table_data = [
        [model, params, cpu_time, gpu_time, speedup]
        for model, (params, cpu_time, gpu_time, speedup) in data.items()
    ]
    return tabulate(table_data, headers, tablefmt="pipe", floatfmt=".2f")

data = {
    "Distil-Whisper-small": (0.17, 2.83, 1.43, 1.98),
    "Distil-Whisper-large": (0.76, 9.76, 0.54, 18.08),
    "Whisper-small": (0.24, 5.98, 0.73, 8.17),
    "Whisper-large": (1.54, 33.16, 2.08, 15.94)
}

print(create_comparison_table(data))