import argparse
import torch
from diffusers import StableDiffusionXLPipeline
from peft import PeftModel
import os

def parse_args():
    parser = argparse.ArgumentParser(description="Inference script for SDXL LoRA.")
    parser.add_argument("--base_model", type=str, default="stabilityai/stable-diffusion-xl-base-1.0")
    parser.add_argument("--lora_path", type=str, default="./sdxl-naruto-lora")
    parser.add_argument("--prompt", type=str, nargs="+", default=[
        "Naruto Uzumaki eating ramen",
        "Bill Gates in Naruto style",
        "A boy with blue eyes in Naruto style"
    ])
    parser.add_argument("--output_dir", type=str, default="inference_outputs")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Load Base Model
    print("Loading base model...")
    pipe = StableDiffusionXLPipeline.from_pretrained(
        args.base_model, 
        torch_dtype=torch.float16, 
        use_safetensors=True, 
        variant="fp16"
    )
    pipe.to("cuda")

    # 2. Generate Baseline Images
    print("Generating baseline images...")
    generator = torch.Generator("cuda").manual_seed(args.seed)
    for i, prompt in enumerate(args.prompt):
        image = pipe(prompt, generator=generator).images[0]
        image.save(os.path.join(args.output_dir, f"baseline_{i}.png"))
        print(f"Saved baseline_{i}.png for prompt: {prompt}")

    # 3. Load LoRA
    print(f"Loading LoRA from {args.lora_path}...")
    pipe.load_lora_weights(args.lora_path)
    
    # 4. Generate Fine-tuned Images
    print("Generating fine-tuned images...")
    generator = torch.Generator("cuda").manual_seed(args.seed)
    for i, prompt in enumerate(args.prompt):
        image = pipe(prompt, generator=generator).images[0]
        image.save(os.path.join(args.output_dir, f"finetuned_{i}.png"))
        print(f"Saved finetuned_{i}.png for prompt: {prompt}")

    print("Inference complete!")

if __name__ == "__main__":
    main()
