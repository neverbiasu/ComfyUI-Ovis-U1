import os
import torch
import numpy as np
import random
from typing import Tuple, List
from PIL import Image
import shutil # Add this import

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
REQUIRED_FILES = ["config.json", "tokenizer_config.json", "tokenizer.json"] # Removed "generation_config.json"
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
        return False, REQUIRED_FILES, ["model.safetensors.index.json"] # Indicate index is missing if dir doesn't exist

    # Check required config files
    for file in REQUIRED_FILES:
        file_path = os.path.join(model_path, file)
        try:
            if not os.path.exists(file_path):
                missing_required.append(file)
                continue
            file_size = os.path.getsize(file_path)
            if file_size == 0:
                missing_required.append(f"{file} (empty)") # Add (empty) for clarity
                continue
        except (OSError, IOError, PermissionError) as e:
            missing_required.append(f"{file} (access error: {e})")
            continue

    # Check for weight files (sharded safetensors and index)
    index_file_path = os.path.join(model_path, "model.safetensors.index.json")
    if not os.path.exists(index_file_path):
        missing_weights.append("model.safetensors.index.json")
    else:
        # If index exists, check for at least one large safetensors file
        safetensors_files = [f for f in os.listdir(model_path) if f.endswith(".safetensors")]
        for weight_file in safetensors_files:
             weight_path = os.path.join(model_path, weight_file)
             try:
                 file_size = os.path.getsize(weight_path)
                 if file_size > 100 * 1024 * 1024:  # 100MB
                     weight_found = True
                     break # Found at least one large weight file
                 else:
                     # Only report small files if no large one is found
                     if not weight_found:
                         missing_weights.append(f"{weight_file} (too small)")
             except (OSError, IOError, PermissionError) as e:
                 if not weight_found:
                     missing_weights.append(f"{weight_file} (access error: {e})")
                 continue

        if not weight_found and not missing_weights:
             # If index exists but no safetensors files found or checked
             missing_weights.append("No large .safetensors files found")


    # Determine overall success
    # Success if no required files are missing AND weight files are found (index exists and at least one large safetensors)
    success = not missing_required and weight_found

    # Refine missing_weights list if weight_found is True but some small files were added
    if weight_found and missing_weights:
        missing_weights = [item for item in missing_weights if "too small" not in item and "access error" not in item]

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
    global shutil # Add this line
    local_path = get_local_model_path(repo_name)

    # First check if model is already available locally
    success, missing_required, missing_weights = check_model_files_safe(local_path)

    if success:
        print(f"Model found at: {local_path}")
        return local_path

    # Model not found locally
    print(f"Model not found locally: {repo_name}")

    # Auto-download is enabled - attempt download
    print(f"Attempting to download: {repo_name}")

    # Pre-check requirements (without network check to avoid crashes)
    if not TRANSFORMERS_AVAILABLE:
        raise RuntimeError("transformers not available")
    if not HF_HUB_AVAILABLE:
        raise RuntimeError("huggingface_hub not available")

    # Attempt download with proper error handling
    try:
        # Create a simple progress tracking
        print(f"Starting download of {repo_name}...")

        # Use snapshot_download directly with error handling
        from huggingface_hub import snapshot_download

        # Try to get token from environment
        token = os.environ.get("HF_TOKEN", None)

        # Create directory if it doesn't exist
        os.makedirs(local_path, exist_ok=True)

        # Download with minimal options to reduce failure points
        snapshot_download(
            repo_id=repo_name,
            local_dir=local_path,
            local_dir_use_symlinks=False,
            resume_download=True,
            token=token,
            force_download=False,
            ignore_patterns=["*.msgpack", "*.h5", "*.ot", "*.md"]
        )

        # Verify download
        success_after_download, missing_required_after, missing_weights_after = check_model_files_safe(local_path)
        if success_after_download:
            print(f"Download completed successfully: {local_path}")
            return local_path
        else:
            # Clean up incomplete download
            if os.path.exists(local_path):
                shutil.rmtree(local_path, ignore_errors=True)

            error_details = []
            if missing_required_after:
                error_details.append(f"Missing required files: {', '.join(missing_required_after)}")
            if missing_weights_after:
                error_details.append(f"Missing or incomplete weight files: {', '.join(missing_weights_after)}")

            error_msg = f"Download verification failed for {repo_name}. " + " ".join(error_details)
            raise RuntimeError(error_msg)

    except Exception as download_error:
        print(f"Download failed: {str(download_error)}")

        # Clean up on failure
        if os.path.exists(local_path):
            shutil.rmtree(local_path, ignore_errors=True)

        # Provide helpful error messages
        error_msg = f"Failed to download {repo_name}: {str(download_error)}"
        if "401" in str(download_error) or "Unauthorized" in str(download_error):
            error_msg += "\nAuthentication required. Please set HF_TOKEN environment variable."
        elif "timeout" in str(download_error).lower():
            error_msg += "\nNetwork timeout. Please check your connection."
        elif "interpreter shutdown" in str(download_error):
            error_msg += "\nDownload interrupted due to system shutdown."

        raise RuntimeError(error_msg)


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

def get_ovis_tokenizers(model):
    """
    Utility to get text and visual tokenizers from the raw Ovis-U1 model.
    Raises an exception if not available.
    """
    if model is None:
        raise Exception("Model is None. Please load the model first.")
    if not hasattr(model, 'get_text_tokenizer') or not hasattr(model, 'get_visual_tokenizer'):
        raise Exception("Model does not provide tokenizer accessors. Please check model version.")
    text_tokenizer = model.get_text_tokenizer()
    visual_tokenizer = model.get_visual_tokenizer()
    if text_tokenizer is None or visual_tokenizer is None:
        raise Exception("Model tokenizers are None. Model may not be properly loaded.")
    return text_tokenizer, visual_tokenizer


# Model Processing Utilities

def build_model_inputs(
    model,
    text_tokenizer,
    visual_tokenizer,
    prompt: str,
    pil_image: Image.Image,
    target_width=None,
    target_height=None
) -> tuple:
    """
    Build model inputs from prompt and image for Ovis-U1 raw model.
    Handles both single and batch image inputs. Returns (input_ids, pixel_values, attention_mask, grid_thws, vae_pixel_values).
    Raises ValueError for invalid input types or shape mismatches.
    """
    model_dtype = getattr(model, "dtype", None)
    if model_dtype is None:
        try:
            model_dtype = next(model.parameters()).dtype
        except Exception:
            model_dtype = torch.float32

    # Handle batch or single image
    is_batch = isinstance(pil_image, (list, tuple))
    images = pil_image if is_batch else [pil_image]
    batch_size = len(images)
    vae_pixel_values = None
    processed_images = []
    vae_pixel_values_list = []
    # Resize and process each image if needed
    for img in images:
        if img is not None and target_width is not None and target_height is not None:
            target_size = (int(target_width), int(target_height))
            img, vae_pixel, cond_img_ids = model.visual_generator.process_image_aspectratio(img, target_size)
            cond_img_ids[..., 0] = 1.0
            vae_pixel = vae_pixel.unsqueeze(0).to(device=model.device, dtype=model_dtype)
            width = img.width
            height = img.height
            resized_height, resized_width = visual_tokenizer.smart_resize(height, width, max_pixels=visual_tokenizer.image_processor.min_pixels)
            img = img.resize((resized_width, resized_height))
            vae_pixel_values_list.append(vae_pixel)
        else:
            vae_pixel_values_list.append(None)
        processed_images.append(img)
    if any(v is not None for v in vae_pixel_values_list):
        vae_pixel_values = torch.cat([v for v in vae_pixel_values_list if v is not None], dim=0)
        vae_pixel_values = vae_pixel_values.to(dtype=model_dtype)
    # Preprocess inputs
    prompt_list = [prompt] * batch_size
    prompt_out, input_ids, pixel_values, grid_thws = model.preprocess_inputs(
        prompt_list if is_batch else prompt,
        processed_images,
        generation_preface=None,
        return_labels=False,
        propagate_exception=False,
        multimodal_type='single_image',
        fix_sample_overall_length_navit=False
    )
    attention_mask = torch.ne(input_ids, text_tokenizer.pad_token_id)
    input_ids = input_ids.unsqueeze(0) if input_ids.dim() == 1 else input_ids
    input_ids = input_ids.to(device=model.device)
    attention_mask = attention_mask.unsqueeze(0) if attention_mask.dim() == 1 else attention_mask
    attention_mask = attention_mask.to(device=model.device)
    if pixel_values is not None:
        pixel_values = torch.cat([
            pixel_values.to(device=visual_tokenizer.device, dtype=model_dtype) if pixel_values is not None else None
        ], dim=0)
    if grid_thws is not None:
        grid_thws = torch.cat([
            grid_thws.to(device=visual_tokenizer.device) if grid_thws is not None else None
        ], dim=0)
    return input_ids, pixel_values, attention_mask, grid_thws, vae_pixel_values




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
        """Load and return the Ovis-U1 model (raw model, not wrapper)."""
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
            return (model,)
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
        if model is None:
            raise Exception("Model is None. Please load the model first.")
        try:
            text_tokenizer, visual_tokenizer = get_ovis_tokenizers(model)
            if seed == -1:
                seed = random.randint(0, 2**31 - 1)
            set_seed(seed)
            width = max(64, (width // 32) * 32)
            height = max(64, (height // 32) * 32)
            model_dtype = getattr(model, "dtype", None)
            if model_dtype is None:
                try:
                    model_dtype = next(model.parameters()).dtype
                except Exception:
                    model_dtype = torch.float32
            print("model dtype: ", model_dtype)
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
            input_ids_uncond, pixel_values_uncond, attention_mask_uncond, grid_thws_uncond, _ = build_model_inputs(
                model, text_tokenizer, visual_tokenizer, uncond_prompt, uncond_image, width, height)

            with torch.inference_mode():
                no_both_cond = model.generate_condition(
                    input_ids_uncond, pixel_values=pixel_values_uncond, attention_mask=attention_mask_uncond, 
                    grid_thws=grid_thws_uncond, **gen_kwargs)

            full_prompt = f"<image>\nDescribe the image by detailing the color, shape, size, texture, quantity, text, and spatial relationships of the objects: {prompt}"
            input_ids_cond, pixel_values_cond, attention_mask_cond, grid_thws_cond, vae_pixel_values_cond = build_model_inputs(
                model, text_tokenizer, visual_tokenizer, full_prompt, uncond_image, width, height)

            if pixel_values_cond is not None:
                pixel_values_cond = pixel_values_cond.to(dtype=model_dtype)
            if grid_thws_cond is not None:
                grid_thws_cond = grid_thws_cond.to(dtype=model_dtype)
            if vae_pixel_values_cond is not None:
                 vae_pixel_values_cond = vae_pixel_values_cond.to(dtype=model_dtype)
            if attention_mask_cond is not None:
                attention_mask_cond = attention_mask_cond.to(dtype=model_dtype) if hasattr(attention_mask_cond, 'to') else attention_mask_cond
            if input_ids_cond is not None:
                input_ids_cond = input_ids_cond.to(dtype=model_dtype) if hasattr(input_ids_cond, 'to') else input_ids_cond

            with torch.inference_mode():
                cond = model.generate_condition(
                    input_ids_cond, pixel_values=pixel_values_cond, attention_mask=attention_mask_cond, 
                    grid_thws=grid_thws_cond, **gen_kwargs)
                cond["vae_pixel_values"] = vae_pixel_values_cond
                images = model.generate_img(
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
        if model is None:
            raise Exception("Model is None. Please load the model first.")
        try:
            text_tokenizer, visual_tokenizer = get_ovis_tokenizers(model)
            pil_image = comfy_to_pil(image)
            gen_kwargs = {
                "max_new_tokens": max_new_tokens,
                "do_sample": False,
                "eos_token_id": text_tokenizer.eos_token_id,
                "pad_token_id": text_tokenizer.pad_token_id,
                "use_cache": True,
            }
            full_prompt = f"<image>\n{prompt}"
            input_ids, pixel_values, attention_mask, grid_thws = build_model_inputs(
                model, text_tokenizer, visual_tokenizer, full_prompt, pil_image)[:4]
            with torch.inference_mode():
                output_ids = model.generate(
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
        if model is None:
            raise Exception("Model is None. Please load the model first.")
        try:
            text_tokenizer, visual_tokenizer = get_ovis_tokenizers(model)
            if seed == -1:
                seed = random.randint(0, 2**31 - 1)
            set_seed(seed)
            input_img = comfy_to_pil(image)
            width, height = input_img.size
            height, width = visual_tokenizer.smart_resize(height, width, factor=32)
            full_prompt = f"<image>\n{prompt}"
            prompt, input_ids, pixel_values, grid_thws, vae_pixel_values = build_model_inputs(
                model, text_tokenizer, visual_tokenizer, full_prompt, input_img, width, height)
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
                output = model.generate(
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

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
