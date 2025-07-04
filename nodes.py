"""
ComfyUI custom nodes for Ovis-U1 multimodal model.

This module provides ComfyUI integration for the Ovis-U1 model with automatic
model downloading, caching, and optimized inference capabilities.
"""

import os
import torch
import numpy as np
import random
from typing import Tuple, List
from PIL import Image

# ComfyUI related imports
from folder_paths import folder_names_and_paths, models_dir as comfy_models_dir, get_folder_paths
from comfy.utils import ProgressBar

# Model-specific imports
try:
    from transformers import AutoModelForCausalLM
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    from huggingface_hub import snapshot_download
    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False

# Register the Ovis model folder in ComfyUI
if "ovis" not in folder_names_and_paths:
    folder_names_and_paths["ovis"] = (
        [os.path.join(comfy_models_dir, "ovis")],
        [".json", ".safetensors", ".bin", ".pt", ".pth"],
    )

# Constants
SUPPORTED_MODELS = ["AIDC-AI/Ovis-U1-3B"]
REQUIRED_FILES = ["config.json", "generation_config.json", "tokenizer_config.json", "tokenizer.json"]
WEIGHT_FILES = ["pytorch_model.bin", "model.safetensors", "pytorch_model.safetensors"]


# Utility Functions

def set_seed(seed: int) -> None:
    """Set random seed for reproducible results."""
    if seed != -1:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def get_local_model_path(repo_name: str) -> str:
    """Get local path for a model repository."""
    model_folder = get_folder_paths("ovis")[0]
    model_name = repo_name.replace("/", "_")
    return os.path.join(model_folder, model_name)


def check_model_files(model_path: str) -> bool:
    """Check if all required model files exist locally."""
    if not os.path.exists(model_path):
        return False
    
    # Check required config files
    for file in REQUIRED_FILES:
        if not os.path.exists(os.path.join(model_path, file)):
            return False
    
    # Check for at least one weight file
    return any(
        os.path.exists(os.path.join(model_path, weight_file)) 
        for weight_file in WEIGHT_FILES
    )


def download_model(repo_name: str, local_path: str, progress_bar: ProgressBar) -> bool:
    """Download model using huggingface_hub."""
    if not HF_HUB_AVAILABLE:
        print("HuggingFace Hub not available. Please install: pip install huggingface_hub")
        return False
    
    try:
        progress_bar.update_absolute(0, 100, f"Downloading {repo_name}...")
        
        snapshot_download(
            repo_id=repo_name,
            local_dir=local_path,
            local_dir_use_symlinks=False,
            resume_download=True,
        )
        
        progress_bar.update_absolute(100, 100, f"Download completed: {repo_name}")
        return True
        
    except Exception as e:
        print(f"Download failed for {repo_name}: {str(e)}")
        return False


def ensure_model_available(repo_name: str) -> str:
    """Ensure model is downloaded and available locally."""
    local_path = get_local_model_path(repo_name)
    
    if check_model_files(local_path):
        return local_path
    
    # Try downloading
    progress_bar = ProgressBar(1)
    if download_model(repo_name, local_path, progress_bar):
        if check_model_files(local_path):
            return local_path
    
    raise RuntimeError(
        f"Failed to download model {repo_name}. "
        f"Please check your internet connection or manually download to {local_path}"
    )


# Image Conversion Utilities

def comfy_to_pil(image_tensor) -> Image.Image:
    """Convert ComfyUI image tensor to PIL Image."""
    i = 255. * image_tensor.cpu().numpy().squeeze()
    return Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))


def pil_to_comfy(pil_image: Image.Image) -> torch.Tensor:
    """Convert PIL Image to ComfyUI image tensor."""
    i = np.array(pil_image).astype(np.float32) / 255.0
    # Ensure 3 channels
    if len(i.shape) == 2:  # Grayscale
        i = np.stack([i] * 3, axis=-1)
    elif i.shape[2] == 4:  # RGBA
        i = i[:, :, :3]  # Remove alpha channel
    
    return torch.from_numpy(i).unsqueeze(0)


def create_blank_image(width: int, height: int) -> Image.Image:
    """Create a blank white image."""
    return Image.new("RGB", (width, height), (255, 255, 255))


# Model Processing Utilities

def build_model_inputs(model, text_tokenizer, visual_tokenizer, prompt: str, pil_image: Image.Image, 
                      target_width=None, target_height=None):
    """Build model inputs from prompt and image."""
    if pil_image is not None and target_width is not None and target_height is not None:
        target_size = (int(target_width), int(target_height))
        pil_image, vae_pixel_values, cond_img_ids = model.visual_generator.process_image_aspectratio(pil_image, target_size)
        cond_img_ids[..., 0] = 1.0
        vae_pixel_values = vae_pixel_values.unsqueeze(0).to(device=model.device)
        width = pil_image.width
        height = pil_image.height
        resized_height, resized_width = visual_tokenizer.smart_resize(height, width, max_pixels=visual_tokenizer.image_processor.min_pixels)
        pil_image = pil_image.resize((resized_width, resized_height))
    else:
        vae_pixel_values = None

    prompt, input_ids, pixel_values, grid_thws = model.preprocess_inputs(
        prompt, 
        [pil_image], 
        generation_preface=None,
        return_labels=False,
        propagate_exception=False,
        multimodal_type='single_image',
        fix_sample_overall_length_navit=False
    )
    
    attention_mask = torch.ne(input_ids, text_tokenizer.pad_token_id)
    input_ids = input_ids.unsqueeze(0).to(device=model.device)
    attention_mask = attention_mask.unsqueeze(0).to(device=model.device)
    
    if pixel_values is not None:
        pixel_values = torch.cat([
            pixel_values.to(device=visual_tokenizer.device, dtype=torch.bfloat16)
        ], dim=0)
    if grid_thws is not None:
        grid_thws = torch.cat([
            grid_thws.to(device=visual_tokenizer.device)
        ], dim=0)
    
    return input_ids, pixel_values, attention_mask, grid_thws, vae_pixel_values


class OvisModelWrapper:
    """Wrapper class for Ovis model with convenient access to tokenizers."""
    
    def __init__(self, model):
        self.model = model
        self.text_tokenizer = model.get_text_tokenizer()
        self.visual_tokenizer = model.get_visual_tokenizer()


# ComfyUI Node Classes

class OvisU1ModelLoader:
    """ComfyUI node for loading Ovis-U1 multimodal model."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_repo_id": (SUPPORTED_MODELS, {"default": SUPPORTED_MODELS[0]}),
                "device": (["auto", "cuda", "cpu"], {"default": "auto"}),
                "dtype": (["bfloat16", "float16", "float32"], {"default": "bfloat16"}),
                "trust_remote_code": ([True, False], {"default": True}),
            }
        }
    
    RETURN_TYPES = ("OVIS_MODEL",)
    FUNCTION = "load_model"
    CATEGORY = "Ovis-U1"
    DESCRIPTION = "Load Ovis-U1 multimodal model with automatic download"

    def load_model(self, model_repo_id: str, device: str, dtype: str, trust_remote_code: bool):
        """Load the Ovis-U1 model with automatic download and caching."""
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers library required. Install: pip install transformers")
        
        try:
            progress_bar = ProgressBar(3)
            
            print(f"Loading Ovis-U1 model: {model_repo_id}")
            progress_bar.update_absolute(0, 3, f"Preparing {model_repo_id}...")
            
            # Ensure model is available locally
            local_model_path = ensure_model_available(model_repo_id)
            progress_bar.update_absolute(1, 3, f"Model files verified")
            
            # Determine device and dtype
            target_device = "cuda" if device == "auto" and torch.cuda.is_available() else device
            torch_dtype = {
                "bfloat16": torch.bfloat16,
                "float16": torch.float16,
                "float32": torch.float32,
            }[dtype]
            
            progress_bar.update_absolute(2, 3, f"Loading on {target_device}...")
            
            # Load model
            model = AutoModelForCausalLM.from_pretrained(
                local_model_path,
                torch_dtype=torch_dtype,
                device_map=target_device if target_device != "auto" else None,
                trust_remote_code=trust_remote_code,
                low_cpu_mem_usage=True,
            )
            
            # Wrap model
            wrapped_model = OvisModelWrapper(model)
            
            progress_bar.update_absolute(3, 3, "Model loaded successfully")
            print(f"Model loaded on {model.device} with dtype {model.dtype}")
            
            return (wrapped_model,)
            
        except Exception as e:
            raise RuntimeError(f"Failed to load model {model_repo_id}: {str(e)}") from e


class OvisU1TextToImage:
    """ComfyUI node for generating images from text using Ovis-U1."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("OVIS_MODEL",),
                "prompt": ("STRING", {"multiline": True, "default": "a cute cat sitting on a windowsill"}),
                "width": ("INT", {"default": 1024, "min": 64, "max": 2048, "step": 64}),
                "height": ("INT", {"default": 1024, "min": 64, "max": 2048, "step": 64}),
                "steps": ("INT", {"default": 50, "min": 1, "max": 200}),
                "txt_cfg": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 20.0, "step": 0.1}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 2**31 - 1}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "text_to_image"
    CATEGORY = "Ovis-U1"
    DESCRIPTION = "Generate image from text using Ovis-U1"
    
    def text_to_image(self, model, prompt, width, height, steps, txt_cfg, seed):
        """Generate an image from text prompt using Ovis-U1 model."""
        try:
            # Set random seed
            if seed == -1:
                seed = random.randint(0, 2**31 - 1)
            set_seed(seed)
            
            # Ensure dimensions are multiples of 32
            width = (width // 32) * 32
            height = (height // 32) * 32
            
            print(f"Generating image: {prompt[:50]}... (size: {width}x{height})")
            
            ovis_model = model.model
            text_tokenizer = model.text_tokenizer
            visual_tokenizer = model.visual_tokenizer
            
            gen_kwargs = {
                "max_new_tokens": 1024,
                "do_sample": False,
                "eos_token_id": text_tokenizer.eos_token_id,
                "pad_token_id": text_tokenizer.pad_token_id,
                "use_cache": True,
                "height": height,
                "width": width,
                "num_steps": steps,
                "seed": seed,
                "img_cfg": 0,
                "txt_cfg": txt_cfg,
            }
            
            # Generate unconditional baseline
            uncond_image = create_blank_image(width, height)
            uncond_prompt = "<image>\nGenerate an image."
            input_ids, pixel_values, attention_mask, grid_thws, _ = build_model_inputs(
                ovis_model, text_tokenizer, visual_tokenizer, uncond_prompt, uncond_image, width, height)
            
            with torch.inference_mode():
                no_both_cond = ovis_model.generate_condition(
                    input_ids, pixel_values=pixel_values, attention_mask=attention_mask, 
                    grid_thws=grid_thws, **gen_kwargs)
            
            # Generate conditional
            full_prompt = f"<image>\nDescribe the image by detailing the color, shape, size, texture, quantity, text, and spatial relationships of the objects: {prompt}"
            input_ids, pixel_values, attention_mask, grid_thws, vae_pixel_values = build_model_inputs(
                ovis_model, text_tokenizer, visual_tokenizer, full_prompt, uncond_image, width, height)
            
            with torch.inference_mode():
                cond = ovis_model.generate_condition(
                    input_ids, pixel_values=pixel_values, attention_mask=attention_mask, 
                    grid_thws=grid_thws, **gen_kwargs)
                cond["vae_pixel_values"] = vae_pixel_values
                images = ovis_model.generate_img(
                    cond=cond, no_both_cond=no_both_cond, no_txt_cond=None, **gen_kwargs)
            
            # Convert to ComfyUI format
            comfy_image = pil_to_comfy(images[0])
            
            print(f"Generated image successfully (seed: {seed})")
            return (comfy_image,)
            
        except Exception as e:
            raise RuntimeError(f"Error in text to image generation: {str(e)}") from e


class OvisU1ImageEdit:
    """ComfyUI node for editing images using text prompts with Ovis-U1."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("OVIS_MODEL",),
                "image": ("IMAGE",),
                "prompt": ("STRING", {"multiline": True, "default": "add a red hat to this cat"}),
                "steps": ("INT", {"default": 50, "min": 1, "max": 200}),
                "txt_cfg": ("FLOAT", {"default": 6.0, "min": 0.0, "max": 20.0, "step": 0.1}),
                "img_cfg": ("FLOAT", {"default": 1.5, "min": 0.0, "max": 20.0, "step": 0.1}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 2**31 - 1}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "edit_image"
    CATEGORY = "Ovis-U1"
    DESCRIPTION = "Edit image using text prompt with Ovis-U1"
    
    def edit_image(self, model, image, prompt, steps, txt_cfg, img_cfg, seed):
        """Edit an image based on text prompt using Ovis-U1 model."""
        try:
            # Set random seed
            if seed == -1:
                seed = random.randint(0, 2**31 - 1)
            set_seed(seed)
            
            # Convert input image format
            input_img = comfy_to_pil(image)
            print(f"Editing image: {prompt[:50]}... (size: {input_img.size})")
            
            ovis_model = model.model
            text_tokenizer = model.text_tokenizer
            visual_tokenizer = model.visual_tokenizer
            
            # Smart resize
            width, height = input_img.size
            height, width = visual_tokenizer.smart_resize(height, width, factor=32)
            
            gen_kwargs = {
                "max_new_tokens": 1024,
                "do_sample": False,
                "eos_token_id": text_tokenizer.eos_token_id,
                "pad_token_id": text_tokenizer.pad_token_id,
                "use_cache": True,
                "height": height,
                "width": width,
                "num_steps": steps,
                "seed": seed,
                "img_cfg": img_cfg,
                "txt_cfg": txt_cfg,
            }
            
            # Generate unconditional baseline
            uncond_image = create_blank_image(width, height)
            uncond_prompt = "<image>\nGenerate an image."
            input_ids, pixel_values, attention_mask, grid_thws, _ = build_model_inputs(
                ovis_model, text_tokenizer, visual_tokenizer, uncond_prompt, uncond_image, width, height)
            
            with torch.inference_mode():
                no_both_cond = ovis_model.generate_condition(
                    input_ids, pixel_values=pixel_values, attention_mask=attention_mask, 
                    grid_thws=grid_thws, **gen_kwargs)
            
            # Generate no-text condition
            input_img_resized = input_img.resize((width, height))
            input_ids, pixel_values, attention_mask, grid_thws, _ = build_model_inputs(
                ovis_model, text_tokenizer, visual_tokenizer, uncond_prompt, input_img_resized, width, height)
            
            with torch.inference_mode():
                no_txt_cond = ovis_model.generate_condition(
                    input_ids, pixel_values=pixel_values, attention_mask=attention_mask, 
                    grid_thws=grid_thws, **gen_kwargs)
            
            # Generate full condition
            full_prompt = f"<image>\n{prompt.strip()}"
            input_ids, pixel_values, attention_mask, grid_thws, vae_pixel_values = build_model_inputs(
                ovis_model, text_tokenizer, visual_tokenizer, full_prompt, input_img_resized, width, height)
            
            with torch.inference_mode():
                cond = ovis_model.generate_condition(
                    input_ids, pixel_values=pixel_values, attention_mask=attention_mask, 
                    grid_thws=grid_thws, **gen_kwargs)
                cond["vae_pixel_values"] = vae_pixel_values
                images = ovis_model.generate_img(
                    cond=cond, no_both_cond=no_both_cond, no_txt_cond=no_txt_cond, **gen_kwargs)
            
            # Convert to ComfyUI format
            comfy_image = pil_to_comfy(images[0])
            
            print(f"Image edited successfully (seed: {seed})")
            return (comfy_image,)
            
        except Exception as e:
            raise RuntimeError(f"Error in image editing: {str(e)}") from e


class OvisU1ImageToText:
    """ComfyUI node for generating text descriptions from images using Ovis-U1."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("OVIS_MODEL",),
                "image": ("IMAGE",),
                "prompt": ("STRING", {"default": "What do you see in this image?", "multiline": True}),
                "max_new_tokens": ("INT", {"default": 4096, "min": 1, "max": 8192}),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    FUNCTION = "image_to_text"
    CATEGORY = "Ovis-U1"
    DESCRIPTION = "Generate text description from image using Ovis-U1"
    
    def image_to_text(self, model, image, prompt, max_new_tokens):
        """Generate text description from image using Ovis-U1 model."""
        try:
            # Convert input image format
            pil_image = comfy_to_pil(image)
            print(f"Analyzing image with prompt: {prompt[:50]}...")
            
            ovis_model = model.model
            text_tokenizer = model.text_tokenizer
            visual_tokenizer = model.visual_tokenizer
            
            gen_kwargs = {
                "max_new_tokens": max_new_tokens,
                "do_sample": False,
                "eos_token_id": text_tokenizer.eos_token_id,
                "pad_token_id": text_tokenizer.pad_token_id,
                "use_cache": True,
            }
            
            # Build inputs
            full_prompt = f"<image>\n{prompt}"
            input_ids, pixel_values, attention_mask, grid_thws = build_model_inputs(
                ovis_model, text_tokenizer, visual_tokenizer, full_prompt, pil_image)[:4]
            
            # Generate text
            with torch.inference_mode():
                output_ids = ovis_model.generate(
                    input_ids, pixel_values=pixel_values, attention_mask=attention_mask, 
                    grid_thws=grid_thws, **gen_kwargs)[0]
                gen_text = text_tokenizer.decode(output_ids, skip_special_tokens=True)
            
            print(f"Generated text: {gen_text[:100]}...")
            return (gen_text,)
            
        except Exception as e:
            raise RuntimeError(f"Error in image to text generation: {str(e)}") from e


# Node registration mappings
NODE_CLASS_MAPPINGS = {
    "OvisU1ModelLoader": OvisU1ModelLoader,
    "OvisU1TextToImage": OvisU1TextToImage,
    "OvisU1ImageEdit": OvisU1ImageEdit,
    "OvisU1ImageToText": OvisU1ImageToText,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OvisU1ModelLoader": "Ovis-U1 Model Loader",
    "OvisU1TextToImage": "Ovis-U1 Text to Image",
    "OvisU1ImageEdit": "Ovis-U1 Image Edit",
    "OvisU1ImageToText": "Ovis-U1 Image to Text",
}

# Export for ComfyUI
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
