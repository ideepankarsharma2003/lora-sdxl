import argparse
import os
import torch
from datasets import load_dataset
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
from diffusers.loaders import LoraLoaderMixin
from diffusers.optimization import get_scheduler
from diffusers.training_utils import compute_snr
from huggingface_hub import login, upload_folder
from peft import LoraConfig, get_peft_model
from torchvision import transforms
from transformers import AutoTokenizer, PretrainedConfig, Trainer, TrainingArguments
import bitsandbytes as bnb
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
import logging
import math
from tqdm.auto import tqdm
import transformers
import diffusers
from dataclasses import dataclass
from typing import Optional, Dict, List, Union, Any
import json
from diffusers.configuration_utils import FrozenDict

# Monkeypatch FrozenDict to be compatible with Trainer
def to_json_string(self):
    return json.dumps(self)

FrozenDict.to_json_string = to_json_string

logger = get_logger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Simple SDXL LoRA training script.")
    parser.add_argument("--pretrained_model_name_or_path", type=str, default="stabilityai/stable-diffusion-xl-base-1.0")
    parser.add_argument("--dataset_name", type=str, default="lambdalabs/naruto-blip-captions")
    parser.add_argument("--output_dir", type=str, default="sdxl-naruto-lora")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--lr_scheduler", type=str, default="constant")
    parser.add_argument("--lr_warmup_steps", type=int, default=0)
    parser.add_argument("--max_train_steps", type=int, default=500)
    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument("--hub_token", type=str, default=None)
    parser.add_argument("--hub_model_id", type=str, default=None)
    parser.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--use_8bit_adam", action="store_true")
    parser.add_argument("--rank", type=int, default=32, help="LoRA rank")
    
    args = parser.parse_args()
    return args

@dataclass
class SDXLDataCollator:
    tokenizer_one: AutoTokenizer
    tokenizer_two: AutoTokenizer
    train_transforms: transforms.Compose
    
    def __call__(self, examples):
        pixel_values = []
        input_ids_one = []
        input_ids_two = []
        
        for example in examples:
            # Image processing
            image = example["image"].convert("RGB")
            pixel_values.append(self.train_transforms(image))
            
            # Text processing
            input_ids_one.append(self.tokenize_captions(example["text"], self.tokenizer_one))
            input_ids_two.append(self.tokenize_captions(example["text"], self.tokenizer_two))
            
        return {
            "pixel_values": torch.stack(pixel_values),
            "input_ids_one": torch.stack(input_ids_one),
            "input_ids_two": torch.stack(input_ids_two),
        }

    def tokenize_captions(self, caption, tokenizer):
        inputs = tokenizer(
            caption, 
            max_length=tokenizer.model_max_length, 
            padding="max_length", 
            truncation=True, 
            return_tensors="pt"
        )
        return inputs.input_ids[0]

class SDXLTrainer(Trainer):
    def __init__(self, *args, text_encoder_one=None, text_encoder_two=None, vae=None, noise_scheduler=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.text_encoder_one = text_encoder_one
        self.text_encoder_two = text_encoder_two
        self.vae = vae
        self.noise_scheduler = noise_scheduler

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # Extract inputs
        pixel_values = inputs["pixel_values"]
        input_ids_one = inputs["input_ids_one"]
        input_ids_two = inputs["input_ids_two"]
        
        # VAE Encoding
        with torch.no_grad():
            latents = self.vae.encode(pixel_values.to(dtype=self.vae.dtype)).latent_dist.sample()
            latents = latents * self.vae.config.scaling_factor

        # Sample noise
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
        timesteps = timesteps.long()

        # Add noise
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

        # Text Encoding
        with torch.no_grad():
            prompt_embeds_list = []
            pooled_prompt_embeds = None
            
            for tokenizer_input_ids, text_encoder in zip(
                [input_ids_one, input_ids_two], 
                [self.text_encoder_one, self.text_encoder_two]
            ):
                output = text_encoder(tokenizer_input_ids, output_hidden_states=True)
                prompt_embeds_list.append(output.hidden_states[-2])
                
                if text_encoder == self.text_encoder_two:
                    pooled_prompt_embeds = output.text_embeds
            
            prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)

        # Time IDs
        # Assuming fixed resolution for simplicity as per original script
        # In a real scenario, this should match the actual crop/size
        resolution = pixel_values.shape[-1] # Assuming square
        add_time_ids = torch.tensor(
            [resolution, resolution, 0, 0, resolution, resolution], 
            device=latents.device, 
            dtype=prompt_embeds.dtype
        ).repeat(bsz, 1)

        added_cond_kwargs = {"text_embeds": pooled_prompt_embeds, "time_ids": add_time_ids}

        # Predict
        # model is the UNet with LoRA
        model_pred = model(noisy_latents, timesteps, prompt_embeds, added_cond_kwargs=added_cond_kwargs).sample

        # Loss
        loss = torch.nn.functional.mse_loss(model_pred.float(), noise.float(), reduction="mean")
        
        return (loss, model_pred) if return_outputs else loss

def main():
    args = parse_args()
    
    # Logging setup
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    if args.seed is not None:
        set_seed(args.seed)

    # Handle HF Login
    if args.push_to_hub:
        if args.hub_token:
            login(token=args.hub_token)
        else:
            logger.warning("Push to hub enabled but no token provided.")

    # Load Tokenizers
    tokenizer_one = AutoTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer", use_fast=False)
    tokenizer_two = AutoTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer_2", use_fast=False)

    # Load Scheduler
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    
    # Load VAE
    vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
    
    # Load UNet
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", torch_dtype=torch.float16, variant="fp16"
    )

    # Freeze VAE and UNet parameters
    vae.requires_grad_(False)
    unet.requires_grad_(False)
    
    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    # LoRA Config
    unet_lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.rank,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )
    
    # Add LoRA to UNet
    unet.add_adapter(unet_lora_config)
    
    # Cast LoRA params to float32
    for param in unet.parameters():
        if param.requires_grad:
            param.data = param.data.to(torch.float32)

    # Text Encoders
    text_encoder_one = transformers.CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", torch_dtype=torch.float16, variant="fp16"
    )
    text_encoder_two = transformers.CLIPTextModelWithProjection.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder_2", torch_dtype=torch.float16, variant="fp16"
    )
    
    text_encoder_one.requires_grad_(False)
    text_encoder_two.requires_grad_(False)

    # Dataset
    dataset = load_dataset(args.dataset_name, split="train")
    
    # Transforms
    train_resize = transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR)
    train_crop = transforms.CenterCrop(args.resolution) 
    train_transforms = transforms.Compose([
        train_resize, 
        train_crop, 
        transforms.ToTensor(), 
        transforms.Normalize([0.5], [0.5])
    ])

    data_collator = SDXLDataCollator(
        tokenizer_one=tokenizer_one,
        tokenizer_two=tokenizer_two,
        train_transforms=train_transforms
    )

    # Training Arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler,
        warmup_steps=args.lr_warmup_steps,
        max_steps=args.max_train_steps,
        fp16=(args.mixed_precision == "fp16"),
        bf16=(args.mixed_precision == "bf16"),
        logging_steps=1,
        save_steps=100,
        push_to_hub=args.push_to_hub,
        hub_token=args.hub_token,
        hub_model_id=args.hub_model_id,
        gradient_checkpointing=args.gradient_checkpointing,
        optim="adamw_bnb_8bit" if args.use_8bit_adam else "adamw_torch",
        remove_unused_columns=False, # Important for custom collator
        report_to="tensorboard",
        logging_dir=os.path.join(args.output_dir, "logs"),
    )

    # Initialize Trainer
    trainer = SDXLTrainer(
        model=unet,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
        text_encoder_one=text_encoder_one,
        text_encoder_two=text_encoder_two,
        vae=vae,
        noise_scheduler=noise_scheduler,
    )

    # Move models to device (Trainer handles model, but we need to handle others or let accelerate handle them if we passed them differently, 
    # but here we pass them as attributes to trainer which puts them on device if they are modules? 
    # Actually Trainer only handles 'model'. We need to ensure others are on the right device.)
    # However, Trainer uses Accelerator internally.
    # We can move them in __init__ or compute_loss using model.device.
    
    # A safer bet for T4 (limited VRAM) is to move them to device manually or let accelerate handle it if we were using it directly.
    # Since we are inside Trainer, we can access trainer.accelerator.
    
    trainer.text_encoder_one.to(trainer.accelerator.device)
    trainer.text_encoder_two.to(trainer.accelerator.device)
    trainer.vae.to(trainer.accelerator.device)

    logger.info("Starting training...")
    trainer.train()
    
    logger.info("Training finished.")
    
    # Save adapter
    trainer.save_model(args.output_dir)
    
    if args.push_to_hub:
        trainer.push_to_hub()

if __name__ == "__main__":
    main()
