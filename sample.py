import torch
import time
import logging
import matplotlib.pyplot as plt
from transformers import pipeline, AutoModelForSpeechSeq2Seq, AutoProcessor

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def get_model_size(model_id):
    model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id)
    total_params = sum(p.numel() for p in model.parameters())
    return total_params / 1e9  # Convert to billions


def transcribe_audio(audio_file, model_id, device, chunk_length_s=30, batch_size=8):
    torch_dtype = torch.float16 if device == "cuda" else torch.float32

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, use_safetensors=True
    )
    model.to(device)

    processor = AutoProcessor.from_pretrained(model_id)

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        max_new_tokens=128,
        chunk_length_s=chunk_length_s,
        batch_size=batch_size,
        torch_dtype=torch_dtype,
        device=device,
    )

    start_time = time.time()
    result = pipe(audio_file, return_timestamps=True)
    end_time = time.time()

    return result["text"], end_time - start_time


def create_comparison_chart(results, model_sizes):
    models = list(results.keys())
    cpu_times = [results[model]['cpu'] for model in models]
    gpu_times = [results[model]['gpu'] for model in models]

    x = range(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(15, 10))
    cpu_bars = ax.bar([i - width / 2 for i in x], cpu_times, width, label='CPU', color='blue')
    gpu_bars = ax.bar([i + width / 2 for i in x], gpu_times, width, label='GPU', color='green')

    ax.set_ylabel('Execution Time (seconds)', fontsize=14)
    ax.set_title('Whisper and Distil-Whisper Models Comparison', fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{model}\n({size:.2f}B params)" for model, size in zip(models, model_sizes)],
                       fontsize=10, rotation=45, ha='right')
    ax.legend(fontsize=12)

    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height,
                    f'{height:.2f}s',
                    ha='center', va='bottom', fontsize=9)

    add_labels(cpu_bars)
    add_labels(gpu_bars)

    # Add speedup text
    for i, model in enumerate(models):
        if gpu_times[i] > 0:  # Avoid division by zero
            speedup = cpu_times[i] / gpu_times[i]
            ax.text(i, max(cpu_times[i], gpu_times[i]), f'GPU Speedup: {speedup:.2f}x',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.savefig('comprehensive_whisper_comparison.png')
    logging.info("Comparison chart saved as 'comprehensive_whisper_comparison.png'")


def main():
    audio_file = "Sample_audio_for_Whisper.mp3"  # Replace with your actual audio file path
    models = {
        'Distil-Whisper-small': "distil-whisper/distil-small.en",
        'Distil-Whisper-large': "distil-whisper/distil-large-v2",
        'Whisper-small': "openai/whisper-small",
        'Whisper-large': "openai/whisper-large-v2"
    }
    results = {model: {'cpu': 0, 'gpu': 0} for model in models}
    model_sizes = []

    for model_name, model_id in models.items():
        # Get model size
        model_size = get_model_size(model_id)
        model_sizes.append(model_size)
        logging.info(f"{model_name} has {model_size:.2f} billion parameters")

        # CPU transcription
        logging.info(f"Starting CPU transcription with {model_name}...")
        _, cpu_time = transcribe_audio(audio_file, model_id, device="cpu")
        results[model_name]['cpu'] = cpu_time
        logging.info(f"{model_name} CPU transcription completed in {cpu_time:.2f} seconds")

        # GPU transcription (if available)
        if torch.cuda.is_available():
            logging.info(f"Starting GPU transcription with {model_name}...")
            _, gpu_time = transcribe_audio(audio_file, model_id, device="cuda")
            results[model_name]['gpu'] = gpu_time
            logging.info(f"{model_name} GPU transcription completed in {gpu_time:.2f} seconds")
        else:
            logging.warning("CUDA is not available. Cannot run GPU transcription.")

    # Create comparison chart
    create_comparison_chart(results, model_sizes)

    # Print detailed results
    print("\nDetailed Results:")
    for model, times in results.items():
        print(f"\n{model} ({model_sizes[list(models.keys()).index(model)]:.2f}B params):")
        print(f"  CPU Time: {times['cpu']:.2f} seconds")
        if times['gpu'] > 0:
            print(f"  GPU Time: {times['gpu']:.2f} seconds")
            print(f"  GPU Speedup: {times['cpu'] / times['gpu']:.2f}x")


if __name__ == "__main__":
    main()