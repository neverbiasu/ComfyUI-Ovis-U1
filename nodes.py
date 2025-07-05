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

# Download configuration - set to False to disable automatic downloads
ENABLE_AUTO_DOWNLOAD = True


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
    model_name = repo_name.split("/")[-1]
    return os.path.join(model_folder, model_name)


def check_model_files_safe(model_path: str) -> Tuple[bool, List[str], List[str]]:
    """Safe model file checking with comprehensive error handling.
    Checks for existence, non-emptiness of required files and sufficient size for weight files.
    Returns a tuple: (success, missing_required_files, missing_weight_files)
    """
    missing_required = []
    missing_weights = []
    weight_found = False

    if not os.path.exists(model_path):
        return False, REQUIRED_FILES, WEIGHT_FILES

    for file in REQUIRED_FILES:
        file_path = os.path.join(model_path, file)
        try:
            if not os.path.exists(file_path):
                missing_required.append(file)
                continue
            file_size = os.path.getsize(file_path)
            if file_size == 0:
                missing_required.append(file)
                continue
        except (OSError, IOError, PermissionError) as e:
            missing_required.append(f"{file} (access error: {e})")
            continue

    for weight_file in WEIGHT_FILES:
        weight_path = os.path.join(model_path, weight_file)
        try:
            if os.path.exists(weight_path):
                file_size = os.path.getsize(weight_path)
                if file_size > 100 * 1024 * 1024:  # 100MB
                    weight_found = True
                    break
                else:
                    missing_weights.append(f"{weight_file} (too small)")
        except (OSError, IOError, PermissionError) as e:
            missing_weights.append(f"{weight_file} (access error: {e})")
            continue

    if not weight_found:
        current_missing_weight_names = [item.split(' ')[0] for item in missing_weights if 'access error' not in item]
        for weight_file in WEIGHT_FILES:
            if weight_file not in current_missing_weight_names and f"{weight_file} (access error" not in ' '.join(missing_weights):
                 missing_weights.append(weight_file)

    success = not missing_required and weight_found

    return success, missing_required, missing_weights


def download_model(repo_name: str, local_path: str, progress_bar: ProgressBar, token: str = None) -> bool:
    """Download model with better error handling and cleanup."""
    if not HF_HUB_AVAILABLE:
        return False

    try:
        if token is None:
            token = os.environ.get("HF_TOKEN", None)

        os.makedirs(local_path, exist_ok=True)

        try:
            snapshot_download(
                repo_id=repo_name,
                local_dir=local_path,
                local_dir_use_symlinks=False,
                resume_download=True,
                token=token,
                force_download=False,
                ignore_patterns=["*.msgpack", "*.h5", "*.ot", "*.md"]
            )
        except Exception:
            return False

        success, missing_required, missing_weights = check_model_files_safe(local_path)
        if success:
            return True
        else:
            return False

    except KeyboardInterrupt:
        return False
    except Exception:
        return False


def pre_check_requirements() -> bool:
    """Pre-check all requirements before attempting download."""
    if not TRANSFORMERS_AVAILABLE:
        return False
    if not HF_HUB_AVAILABLE:
        return False
    return True


def ensure_model_available(repo_name: str) -> str:
    """Ensure model is available, with controlled auto-download."""
    local_path = get_local_model_path(repo_name)

    success, missing_required, missing_weights = check_model_files_safe(local_path)

    if success:
        return local_path

    if not ENABLE_AUTO_DOWNLOAD:
        raise RuntimeError(f"Model not available and auto-download disabled: {repo_name}")

    if not TRANSFORMERS_AVAILABLE:
        raise RuntimeError("transformers not available")
    if not HF_HUB_AVAILABLE:
        raise RuntimeError("huggingface_hub not available")

    try:
        progress_bar = ProgressBar(100)
        ok = download_model(repo_name, local_path, progress_bar)
        success_after_download, _, _ = check_model_files_safe(local_path)
        if ok and success_after_download:
            return local_path
        else:
            raise RuntimeError(f"Model not available after download attempt: {repo_name}")
    except Exception as download_error:
        raise RuntimeError(f"Failed to download {repo_name}: {str(download_error)}")


def comfy_to_pil(image_tensor) -> Image.Image:
    """Convert ComfyUI image tensor to PIL Image."""
    if isinstance(image_tensor, str):
        print("Warning: Expected tensor, got string. Creating blank image.")
        return create_blank_image(256, 256)
    
    try:
        if hasattr(image_tensor, 'cpu'):
            image_tensor = image_tensor.cpu()
        
        # Handle different tensor shapes
        if len(image_tensor.shape) == 4:  # Batch dimension
            image_tensor = image_tensor.squeeze(0)
        elif len(image_tensor.shape) == 2:  # Grayscale without channel
            image_tensor = image_tensor.unsqueeze(-1)
        
        # Convert to numpy and scale to 0-255
        i = 255. * image_tensor.numpy()
        i = np.clip(i, 0, 255).astype(np.uint8)
        
        # Handle different channel configurations
        if i.shape[-1] == 1:  # Grayscale
            i = i.squeeze(-1)
            return Image.fromarray(i, mode='L')
        elif i.shape[-1] == 3:  # RGB
            return Image.fromarray(i, mode='RGB')
        elif i.shape[-1] == 4:  # RGBA
            return Image.fromarray(i, mode='RGBA')
        else:
            print(f"Warning: Unsupported image tensor shape: {i.shape}. Creating blank image.")
            return create_blank_image(256, 256)
    except Exception as e:
        print(f"Error converting tensor to PIL image: {str(e)}. Creating blank image.")
        return create_blank_image(256, 256)


def pil_to_comfy(pil_image: Image.Image) -> torch.Tensor:
    """Convert PIL Image to ComfyUI image tensor."""
    if not isinstance(pil_image, Image.Image):
        print(f"Warning: Expected PIL Image, got: {str(type(pil_image))}. Creating blank tensor.")
        # Create a blank tensor as fallback
        blank_array = np.ones((256, 256, 3), dtype=np.float32)
        return torch.from_numpy(blank_array).unsqueeze(0)
    
    try:
        # Convert to RGB if needed
        if pil_image.mode == 'RGBA':
            # Create white background and paste RGBA image
            background = Image.new('RGB', pil_image.size, (255, 255, 255))
            background.paste(pil_image, mask=pil_image.split()[-1])
            pil_image = background
        elif pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        
        # Convert to numpy array and normalize to 0-1
        i = np.array(pil_image).astype(np.float32) / 255.0
        
        # Add batch dimension
        return torch.from_numpy(i).unsqueeze(0)
    except Exception as e:
        print(f"Error converting PIL image to tensor: {str(e)}. Creating blank tensor.")
        blank_array = np.ones((256, 256, 3), dtype=np.float32)
        return torch.from_numpy(blank_array).unsqueeze(0)


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
    """Ultra-safe wrapper class for Ovis model."""
    
    def __init__(self, model):
        self.model = None
        self.text_tokenizer = None
        self.visual_tokenizer = None
        
        try:
            if model is None:
                return
            if not hasattr(model, 'get_text_tokenizer'):
                return
            if not hasattr(model, 'get_visual_tokenizer'):
                return
            self.model = model
            try:
                self.text_tokenizer = model.get_text_tokenizer()
            except Exception:
                return
            try:
                self.visual_tokenizer = model.get_visual_tokenizer()
            except Exception:
                return
            if self.text_tokenizer is None or self.visual_tokenizer is None:
                self.model = None
        except Exception:
            self.model = None
    
    def is_valid(self):
        """Check if the wrapper has valid model components."""
        try:
            return (self.model is not None and 
                    self.text_tokenizer is not None and 
                    self.visual_tokenizer is not None)
        except Exception:
            return False


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
        """Strict model loading with proper error handling."""
        try:
            if not TRANSFORMERS_AVAILABLE:
                raise ImportError("transformers library required. Install: pip install transformers")
            local_model_path = ensure_model_available(model_repo_id)
            if local_model_path is None:
                raise Exception(f"Model not available: {model_repo_id}")
            if device == "auto":
                target_device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                target_device = device
            dtype_map = {
                "bfloat16": torch.bfloat16,
                "float16": torch.float16,
                "float32": torch.float32,
            }
            torch_dtype = dtype_map.get(dtype, torch.float32)
            model = AutoModelForCausalLM.from_pretrained(
                local_model_path,
                torch_dtype=torch_dtype,
                device_map=target_device,
                trust_remote_code=trust_remote_code,
                low_cpu_mem_usage=True,
            )
            if model is None:
                raise Exception("Model loading returned None")
            wrapped_model = OvisModelWrapper(model)
            if not wrapped_model.is_valid():
                raise Exception("Model wrapper validation failed - model is not compatible")
            return (wrapped_model,)
        except ImportError as e:
            error_msg = f"Import error: {str(e)}"
            raise Exception(error_msg) from e
        except FileNotFoundError as e:
            error_msg = f"File not found: {str(e)}"
            raise Exception(error_msg) from e
        except torch.cuda.OutOfMemoryError as e:
            error_msg = "GPU out of memory. Try using CPU."
            raise Exception(error_msg) from e
        except Exception as e:
            error_msg = f"Model loading failed: {type(e).__name__}: {str(e)}"
            raise Exception(error_msg) from e


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
        # Strict model validation
        if model is None:
            raise Exception("Model is None. Please load the model first.")
        if not hasattr(model, 'is_valid') or not model.is_valid():
            raise Exception("Model is not loaded or invalid. Please load the model first.")
        if not hasattr(model, 'model') or model.model is None:
            raise Exception("Model wrapper contains no actual model. Please reload the model.")
        
        try:
            if seed == -1:
                seed = random.randint(0, 2**31 - 1)
            set_seed(seed)
            width = max(64, (width // 32) * 32)
            height = max(64, (height // 32) * 32)
            ovis_model = model.model
            text_tokenizer = model.text_tokenizer
            visual_tokenizer = model.visual_tokenizer
            if text_tokenizer is None or visual_tokenizer is None:
                raise Exception("Model tokenizers are None. Model may not be properly loaded.")
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
            uncond_image = create_blank_image(width, height)
            uncond_prompt = "<image>\nGenerate an image."
            input_ids, pixel_values, attention_mask, grid_thws, _ = build_model_inputs(
                ovis_model, text_tokenizer, visual_tokenizer, uncond_prompt, uncond_image, width, height)
            with torch.inference_mode():
                no_both_cond = ovis_model.generate_condition(
                    input_ids, pixel_values=pixel_values, attention_mask=attention_mask, 
                    grid_thws=grid_thws, **gen_kwargs)
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
            comfy_image = pil_to_comfy(images[0])
            return (comfy_image,)
        except Exception as e:
            error_msg = f"Error in text to image generation: {str(e)}"
            raise Exception(error_msg) from e


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
        # Strict model validation
        if model is None:
            raise Exception("Model is None. Please load the model first.")
        if not hasattr(model, 'is_valid') or not model.is_valid():
            raise Exception("Model is not loaded or invalid. Please load the model first.")
        if not hasattr(model, 'model') or model.model is None:
            raise Exception("Model wrapper contains no actual model. Please reload the model.")

        try:
            pil_image = comfy_to_pil(image)
            ovis_model = model.model
            text_tokenizer = model.text_tokenizer
            visual_tokenizer = model.visual_tokenizer
            if text_tokenizer is None or visual_tokenizer is None:
                raise Exception("Model tokenizers are None. Model may not be properly loaded.")
            gen_kwargs = {
                "max_new_tokens": max_new_tokens,
                "do_sample": False,
                "eos_token_id": text_tokenizer.eos_token_id,
                "pad_token_id": text_tokenizer.pad_token_id,
                "use_cache": True,
            }
            full_prompt = f"<image>\n{prompt}"
            input_ids, pixel_values, attention_mask, grid_thws = build_model_inputs(
                ovis_model, text_tokenizer, visual_tokenizer, full_prompt, pil_image)[:4]
            with torch.inference_mode():
                output_ids = ovis_model.generate(
                    input_ids, pixel_values=pixel_values, attention_mask=attention_mask, 
                    grid_thws=grid_thws, **gen_kwargs)[0]
                gen_text = text_tokenizer.decode(output_ids, skip_special_tokens=True)
            return (gen_text,)
        except Exception as e:
            error_msg = f"Error: Failed to analyze image - {str(e)}"
            raise Exception(error_msg) from e


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
        # Strict model validation
        if model is None:
            raise Exception("Model is None. Please load the model first.")
        if not hasattr(model, 'is_valid') or not model.is_valid():
            raise Exception("Model is not loaded or invalid. Please load the model first.")
        if not hasattr(model, 'model') or model.model is None:
            raise Exception("Model wrapper contains no actual model. Please reload the model.")
        
        try:
            if seed == -1:
                seed = random.randint(0, 2**31 - 1)
            set_seed(seed)
            input_img = comfy_to_pil(image)
            ovis_model = model.model
            text_tokenizer = model.text_tokenizer
            visual_tokenizer = model.visual_tokenizer
            if text_tokenizer is None or visual_tokenizer is None:
                raise Exception("Model tokenizers are None. Model may not be properly loaded.")
            width, height = input_img.size
            height, width = visual_tokenizer.smart_resize(height, width, factor=32)
            full_prompt = f"<image>\n{prompt}"
            prompt, input_ids, pixel_values, grid_thws, vae_pixel_values = build_model_inputs(
                ovis_model, text_tokenizer, visual_tokenizer, full_prompt, input_img, width, height)
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
            with torch.inference_mode():
                output = ovis_model.generate(
                    input_ids, pixel_values=pixel_values, attention_mask=attention_mask,
                    grid_thws=grid_thws, **gen_kwargs)
                images = output.images
                if len(images) == 0:
                    raise Exception("No images found in generation output")
                gen_image = images[0]
            comfy_image = pil_to_comfy(gen_image)
            return (comfy_image,)
        except Exception as e:
            error_msg = f"Error in image editing: {str(e)}"
            raise Exception(error_msg) from e


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
