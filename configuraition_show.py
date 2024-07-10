import sys
import platform
import torch
import psutil
import cpuinfo


def get_size(bytes, suffix="B"):
    """
    Scale bytes to its proper format
    e.g:
        1253656 => '1.20MB'
        1253656678 => '1.17GB'
    """
    factor = 1024
    for unit in ["", "K", "M", "G", "T", "P"]:
        if bytes < factor:
            return f"{bytes:.2f}{unit}{suffix}"
        bytes /= factor


def display_config():
    print("System Configuration:")
    print(f"OS: {platform.system()} {platform.release()}")
    print(f"Python version: {sys.version}")

    print("\nCPU Info:")
    cpu_info = cpuinfo.get_cpu_info()
    print(f"Processor: {cpu_info['brand_raw']}")
    print(f"Architecture: {cpu_info['arch']}")
    print(f"Cores: {psutil.cpu_count(logical=False)} (Physical), {psutil.cpu_count(logical=True)} (Logical)")

    print("\nMemory Info:")
    svmem = psutil.virtual_memory()
    print(f"Total: {get_size(svmem.total)}")
    print(f"Available: {get_size(svmem.available)}")
    print(f"Used: {get_size(svmem.used)} ({svmem.percent}%)")

    print("\nPyTorch / CUDA Info:")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"cuDNN version: {torch.backends.cudnn.version()}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"    Memory: {get_size(torch.cuda.get_device_properties(i).total_memory)}")
    else:
        print("CUDA is not available on this system.")


if __name__ == "__main__":
    display_config()