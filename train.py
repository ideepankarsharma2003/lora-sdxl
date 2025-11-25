import argparse
import os
import torch
import gc
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

logger = logging.getLogger(__name__)

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

def import_model_class_from_model_name_or_path(pretrained_model_name_or_path: str, revision: str):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=revision,
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "RobertaSeriesModelWithTransformation":
        from diffusers.pipelines.alt_diffusion.modeling_roberta_series import RobertaSeriesModelWithTransformation

        return RobertaSeriesModelWithTransformation
    else:
        raise ValueError(f"{model_class} is not supported.")

def tokenize_captions(captions, tokenizer):
    inputs = tokenizer(
        captions, 
        max_length=tokenizer.model_max_length, 
        padding="max_length", 
        truncation=True, 
        return_tensors="pt"
    )
    return inputs.input_ids

@torch.no_grad()
def prepare_train_dataset(dataset, tokenizer_one, tokenizer_two, text_encoder_one, text_encoder_two, vae, args):
    # Pre-compute latents and embeddings
    # This function will return a new dataset with "model_input", "prompt_embeds", "pooled_prompt_embeds", "time_ids"
    
    processed_data = []
    
    train_resize = transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR)
    train_crop = transforms.CenterCrop(args.resolution) 
    train_transforms = transforms.Compose([
        train_resize, 
        train_crop, 
        transforms.ToTensor(), 
        transforms.Normalize([0.5], [0.5])
    ])
    
    logger.info("Pre-computing latents and embeddings...")
    
    # Move models to GPU for processing
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae.to(device)
    text_encoder_one.to(device)
    text_encoder_two.to(device)
    
    for example in tqdm(dataset):
        # 1. Image -> Latents
        image = example["image"].convert("RGB")
        pixel_values = train_transforms(image).unsqueeze(0).to(device, dtype=vae.dtype)
        model_input = vae.encode(pixel_values).latent_dist.sample()
        model_input = model_input * vae.config.scaling_factor
        model_input = model_input.squeeze(0).cpu() # Move back to CPU to save VRAM
        
        # 2. Text -> Embeddings
        caption = example["text"]
        
        # Tokenize
        input_ids_one = tokenize_captions([caption], tokenizer_one).to(device)
        input_ids_two = tokenize_captions([caption], tokenizer_two).to(device)
        
        # Encode
        prompt_embeds_list = []
        
        # Text Encoder 1
        output_one = text_encoder_one(input_ids_one, output_hidden_states=True)
        prompt_embeds_list.append(output_one.hidden_states[-2])
        
        # Text Encoder 2
        output_two = text_encoder_two(input_ids_two, output_hidden_states=True)
        prompt_embeds_list.append(output_two.hidden_states[-2])
        pooled_prompt_embeds = output_two.text_embeds
        
        prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
        
        # 3. Time IDs
        # Assuming fixed resolution
        original_size = (args.resolution, args.resolution)
        target_size = (args.resolution, args.resolution)
        crops_coords_top_left = (0, 0)
        
        add_time_ids = list(original_size + crops_coords_top_left + target_size)
        add_time_ids = torch.tensor([add_time_ids])
        
        processed_data.append({
            "model_input": model_input,
            "prompt_embeds": prompt_embeds.squeeze(0).cpu(),
            "pooled_prompt_embeds": pooled_prompt_embeds.squeeze(0).cpu(),
            "time_ids": add_time_ids.squeeze(0).cpu()
        })
        
    return processed_data

@dataclass
class SDXLDataCollator:
    def __call__(self, examples):
        model_input = torch.stack([x["model_input"] for x in examples])
        prompt_embeds = torch.stack([x["prompt_embeds"] for x in examples])
        pooled_prompt_embeds = torch.stack([x["pooled_prompt_embeds"] for x in examples])
        time_ids = torch.stack([x["time_ids"] for x in examples])
        
        return {
            "model_input": model_input,
            "prompt_embeds": prompt_embeds,
            "pooled_prompt_embeds": pooled_prompt_embeds,
            "time_ids": time_ids,
        }

class SDXLTrainer(Trainer):
    def __init__(self, *args, noise_scheduler=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.noise_scheduler = noise_scheduler

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # Inputs are already latents and embeddings
        latents = inputs["model_input"]
        prompt_embeds = inputs["prompt_embeds"]
        pooled_prompt_embeds = inputs["pooled_prompt_embeds"]
        add_time_ids = inputs["time_ids"]
        
        # Sample noise
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
        timesteps = timesteps.long()

        # Add noise
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

        # Prepare added_cond_kwargs
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

    # 1. Load Tokenizers and Models for Preparation
    logger.info("Loading models for data preparation...")
    tokenizer_one = AutoTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer", use_fast=False)
    tokenizer_two = AutoTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer_2", use_fast=False)
    
    text_encoder_cls_one = import_model_class_from_model_name_or_path(args.pretrained_model_name_or_path, revision=None)
    text_encoder_cls_two = import_model_class_from_model_name_or_path(args.pretrained_model_name_or_path, revision=None) # Actually usually CLIPTextModelWithProjection

    # Load VAE
    vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
    
    # Load Text Encoders
    text_encoder_one = text_encoder_cls_one.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", torch_dtype=torch.float16, variant="fp16"
    )
    text_encoder_two = transformers.CLIPTextModelWithProjection.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder_2", torch_dtype=torch.float16, variant="fp16"
    )
    
    # Dataset
    dataset = load_dataset(args.dataset_name, split="train")
    
    # 2. Pre-compute Data
    train_dataset = prepare_train_dataset(dataset, tokenizer_one, tokenizer_two, text_encoder_one, text_encoder_two, vae, args)
    
    # 3. Unload Models and Clear Cache
    logger.info("Unloading preparation models...")
    del tokenizer_one, tokenizer_two, text_encoder_one, text_encoder_two, vae
    gc.collect()
    torch.cuda.empty_cache()
    
    # 4. Load UNet for Training
    logger.info("Loading UNet for training...")
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", torch_dtype=torch.float16, variant="fp16"
    )
    
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

    # Load Scheduler
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")

    data_collator = SDXLDataCollator()

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
        gradient_checkpointing=False,
        optim="adamw_bnb_8bit" if args.use_8bit_adam else "adamw_torch",
        remove_unused_columns=False,
        report_to="tensorboard",
        logging_dir=os.path.join(args.output_dir, "logs"),
    )

    # Initialize Trainer
    trainer = SDXLTrainer(
        model=unet,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        noise_scheduler=noise_scheduler,
    )

    logger.info("Starting training...")
    trainer.train()
    
    logger.info("Training finished.")
    
    # Save adapter
    trainer.save_model(args.output_dir)
    
    if args.push_to_hub:
        trainer.push_to_hub()

if __name__ == "__main__":
    main()
