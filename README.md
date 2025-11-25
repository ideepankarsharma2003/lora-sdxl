# SDXL Fine-tuning on Naruto Dataset (T4 GPU Optimized)

This repository contains code to fine-tune Stable Diffusion XL (SDXL) 1.0 on the Naruto BLIP captions dataset using Low-Rank Adaptation (LoRA). The project is specifically optimized to run on a Google Colab T4 GPU (16GB VRAM).

## Project Structure

- `train.py`: The main training script using `accelerate`, `diffusers`, and `peft`.
- `inference.py`: A script to generate images using the base model and the fine-tuned LoRA for comparison.
- `requirements.txt`: List of Python dependencies.

## Hardware Constraints & Optimization Strategy

Training SDXL (approx. 6.6B parameters) on a single T4 GPU (16GB VRAM) is challenging because the model weights, gradients, and optimizer states easily exceed the available memory at standard precision.

To solve this, I implemented the following optimizations:

1.  **Low-Rank Adaptation (LoRA)**:
    -   Instead of fine-tuning the full UNet, we freeze the pre-trained weights and inject trainable rank-decomposition matrices (rank=32) into the attention layers.
    -   This reduces the number of trainable parameters from billions to a few million, drastically lowering memory usage for gradients and optimizer states.

2.  **Precision Management (Mixed Precision fp16)**:
    -   Training is performed in `fp16` (half-precision).
    -   Weights are kept in half-precision where possible.
    -   This effectively halves the memory required for model weights and activations compared to fp32.

3.  **Gradient Checkpointing**:
    -   Enabled on the UNet.
    -   This technique trades compute for memory by not storing all intermediate activations during the forward pass. Instead, they are recomputed during the backward pass. This is crucial for fitting larger batch sizes or resolutions.

4.  **8-bit Adam Optimizer**:
    -   Used `bitsandbytes` `AdamW8bit`.
    -   Standard AdamW stores 32-bit states for every parameter (momentum and variance). 8-bit Adam compresses these states, significantly reducing the memory footprint of the optimizer.

5.  **Frozen Text Encoders & VAE**:
    -   The VAE and both Text Encoders (CLIP ViT-L and OpenCLIP ViT-bigG) are kept frozen.
    -   We only train the UNet (via LoRA).
    -   This avoids storing gradients and optimizer states for these large components.

6.  **Gradient Accumulation**:
    -   To achieve an effective batch size suitable for training without OOM, I use a physical batch size of 1 and accumulate gradients over 4 steps (effective batch size = 4).

7.  **Pre-computation of Latents & Embeddings**:
    -   **Crucial for T4**: Instead of encoding images and text on-the-fly during training (which keeps VAE and Text Encoders in VRAM), I pre-compute all latents and embeddings before the training loop starts.
    -   This allows me to **unload** the VAE and Text Encoders from memory, freeing up significant VRAM (approx. 4GB+) for the UNet and optimizer states.
    -   *Note*: This adds a small startup time to process the dataset but enables training on 16GB VRAM without OOM.

## Setup & Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/ideepankarsharma2003/lora-sdxl.git
    cd lora-sdxl
    ```

2.  **Install dependencies**:
    You can use the provided setup script:
    ```bash
    chmod +x setup.sh
    ./setup.sh
    ```
    Or install manually:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Hugging Face Login** (Optional, for pushing model):
    ```bash
    huggingface-cli login
    ```

## Usage

### Training

Run the training script. By default, it uses the settings optimized for T4 (Resolution 1024, Batch Size 1, Gradient Accumulation 4).

```bash
accelerate launch --mixed_precision="fp16" train.py \
  --pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0" \
  --dataset_name="lambdalabs/naruto-blip-captions" \
  --output_dir="sdxl-naruto-lora" \
  --resolution=1024 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --max_train_steps=500 \
  --learning_rate=1e-4 \
  --mixed_precision="fp16" \
  --gradient_checkpointing \
  --use_8bit_adam \
  --rank=32
```

To push to Hugging Face Hub automatically, add:
```bash
  --push_to_hub \
  --hub_model_id="your-username/sdxl-naruto-lora" \
  --hub_token="your_token"
```

### Inference

Run the inference script to generate comparisons.

```bash
python inference.py \
  --base_model="stabilityai/stable-diffusion-xl-base-1.0" \
  --lora_path="./sdxl-naruto-lora" \
  --prompt "Naruto Uzumaki eating ramen" "Bill Gates in Naruto style" \
  --output_dir="inference_outputs"
```

## Results

The script will save the fine-tuned LoRA weights in the `sdxl-naruto-lora` directory. The inference script will output `baseline_*.png` and `finetuned_*.png` images in `inference_outputs` for side-by-side comparison.

## References

- [Stable Diffusion XL Paper](https://arxiv.org/abs/2307.01952)
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- [Hugging Face Diffusers Documentation](https://huggingface.co/docs/diffusers/index)
