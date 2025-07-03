import os
import torch
import numpy as np
from PIL import Image
from transformers import AutoModelForCausalLM

# ComfyUI related imports
import comfy.model_management as model_management


def comfy_to_pil(image_tensor):
    """Convert ComfyUI image tensor to PIL Image.
    
    Args:
        image_tensor: ComfyUI image tensor with shape [1,H,W,3] and values in range [0,1]
        
    Returns:
        PIL.Image: RGB PIL Image
    """
    # ComfyUI image format: batch, height, width, channels (0-1 float)
    i = 255. * image_tensor.cpu().numpy().squeeze()
    img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
    return img


def pil_to_comfy(pil_image):
    """Convert PIL Image to ComfyUI image tensor.
    
    Args:
        pil_image: PIL Image (RGB, RGBA, or grayscale)
        
    Returns:
        torch.Tensor: ComfyUI image tensor with shape [1,H,W,3] and values in range [0,1]
    """
    # PIL -> numpy -> tensor -> ComfyUI format
    i = np.array(pil_image).astype(np.float32) / 255.0
    # Ensure 3 channels
    if len(i.shape) == 2:  # Grayscale
        i = np.stack([i, i, i], axis=-1)
    elif i.shape[2] == 4:  # RGBA
        i = i[:, :, :3]
    # Add batch dimension
    return torch.from_numpy(i)[None,]


def load_blank_image(width, height):
    """Create a blank white image.
    
    Args:
        width: Image width
        height: Image height
        
    Returns:
        PIL.Image: White RGB image
    """
    return Image.new("RGB", (width, height), (255, 255, 255)).convert('RGB')


def build_inputs(model, text_tokenizer, visual_tokenizer, prompt, pil_image, target_width=None, target_height=None):
    """Build model inputs from prompt and image.
    
    Args:
        model: Ovis model instance
        text_tokenizer: Text tokenizer
        visual_tokenizer: Visual tokenizer
        prompt: Text prompt
        pil_image: PIL Image
        target_width: Target image width
        target_height: Target image height
        
    Returns:
        tuple: (input_ids, pixel_values, attention_mask, grid_thws, vae_pixel_values)
    """
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
        cond_img_ids = None

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
            pixel_values.to(device=visual_tokenizer.device, dtype=torch.bfloat16) if pixel_values is not None else None
        ], dim=0)
    if grid_thws is not None:
        grid_thws = torch.cat([
            grid_thws.to(device=visual_tokenizer.device) if grid_thws is not None else None
        ], dim=0)
    
    return input_ids, pixel_values, attention_mask, grid_thws, vae_pixel_values


class OvisModelWrapper:
    """Wrapper class for Ovis model with convenient access to tokenizers."""
    
    def __init__(self, model):
        """Initialize the wrapper.
        
        Args:
            model: Loaded Ovis model instance
        """
        self.model = model
        self.text_tokenizer = model.get_text_tokenizer()
        self.visual_tokenizer = model.get_visual_tokenizer()


class OvisU1ModelLoader:
    """ComfyUI node for loading Ovis-U1 multimodal model."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_path": ("STRING", {"default": "AIDC-AI/Ovis-U1-3B"}),
                "device": (["auto", "cuda", "cpu"], {"default": "auto"}),
                "dtype": (["bfloat16", "float16", "float32"], {"default": "bfloat16"}),
            }
        }
    
    RETURN_TYPES = ("OVIS_MODEL",)
    FUNCTION = "load_model"
    CATEGORY = "Ovis-U1"
    DESCRIPTION = "Load Ovis-U1 multimodal model"
    
    def load_model(self, model_path, device, dtype):
        """Load the Ovis-U1 model with specified configuration.
        
        Args:
            model_path: Path or identifier of the model to load
            device: Device to load the model on ('auto', 'cuda', 'cpu')
            dtype: Data type for model weights ('bfloat16', 'float16', 'float32')
            
        Returns:
            tuple: (OvisModelWrapper,) containing the loaded model
        """
        try:
            # Set device
            if device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            
            # Set data type
            torch_dtype_map = {
                "bfloat16": torch.bfloat16,
                "float16": torch.float16,
                "float32": torch.float32
            }
            torch_dtype = torch_dtype_map[dtype]
            
            print(f"Loading Ovis-U1 model from {model_path}")
            print(f"Device: {device}, Dtype: {dtype}")
            
            # Load model
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch_dtype,
                trust_remote_code=True
            )
            
            # Move to specified device
            model = model.eval().to(device)
            if torch_dtype != torch.float32:
                model = model.to(torch_dtype)
            
            # Wrap model
            wrapped_model = OvisModelWrapper(model)
            
            print("Ovis-U1 model loaded successfully")
            return (wrapped_model,)
            
        except Exception as e:
            print(f"Error loading Ovis-U1 model: {str(e)}")
            raise e


class OvisU1TextToImage:
    """ComfyUI node for generating images from text using Ovis-U1."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("OVIS_MODEL",),
                "prompt": ("STRING", {"multiline": True, "default": "a cute cat"}),
                "width": ("INT", {"default": 1024, "min": 64, "max": 2048, "step": 64}),
                "height": ("INT", {"default": 1024, "min": 64, "max": 2048, "step": 64}),
                "steps": ("INT", {"default": 50, "min": 1, "max": 200}),
                "txt_cfg": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 20.0, "step": 0.1}),
                "seed": ("INT", {"default": 42}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "text_to_image"
    CATEGORY = "Ovis-U1"
    DESCRIPTION = "Generate image from text using Ovis-U1"
    
    def text_to_image(self, model, prompt, width, height, steps, txt_cfg, seed):
        """Generate an image from text prompt using Ovis-U1 model.
        
        Args:
            model: OvisModelWrapper instance
            prompt: Text description of the image to generate
            width: Image width in pixels (will be rounded to multiple of 32)
            height: Image height in pixels (will be rounded to multiple of 32)
            steps: Number of diffusion steps
            txt_cfg: Text guidance scale
            seed: Random seed for generation
            
        Returns:
            tuple: (ComfyUI image tensor,)
        """
        try:
            # Ensure dimensions are multiples of 32
            width = (width // 32) * 32
            height = (height // 32) * 32
            
            ovis_model = model.model
            text_tokenizer = model.text_tokenizer
            visual_tokenizer = model.visual_tokenizer
            
            gen_kwargs = dict(
                max_new_tokens=1024,
                do_sample=False,
                top_p=None,
                top_k=None,
                temperature=None,
                repetition_penalty=None,
                eos_token_id=text_tokenizer.eos_token_id,
                pad_token_id=text_tokenizer.pad_token_id,
                use_cache=True,
                height=height,
                width=width,
                num_steps=steps,
                seed=seed,
                img_cfg=0,
                txt_cfg=txt_cfg,
            )
            
            # Generate unconditional baseline
            uncond_image = load_blank_image(width, height)
            uncond_prompt = "<image>\nGenerate an image."
            input_ids, pixel_values, attention_mask, grid_thws, _ = build_inputs(
                ovis_model, text_tokenizer, visual_tokenizer, uncond_prompt, uncond_image, width, height)
            
            with torch.inference_mode():
                no_both_cond = ovis_model.generate_condition(
                    input_ids, pixel_values=pixel_values, attention_mask=attention_mask, 
                    grid_thws=grid_thws, **gen_kwargs)
            
            # Generate conditional
            prompt = "<image>\nDescribe the image by detailing the color, shape, size, texture, quantity, text, and spatial relationships of the objects:" + prompt
            no_txt_cond = None
            input_ids, pixel_values, attention_mask, grid_thws, vae_pixel_values = build_inputs(
                ovis_model, text_tokenizer, visual_tokenizer, prompt, uncond_image, width, height)
            
            with torch.inference_mode():
                cond = ovis_model.generate_condition(
                    input_ids, pixel_values=pixel_values, attention_mask=attention_mask, 
                    grid_thws=grid_thws, **gen_kwargs)
                cond["vae_pixel_values"] = vae_pixel_values
                images = ovis_model.generate_img(
                    cond=cond, no_both_cond=no_both_cond, no_txt_cond=no_txt_cond, **gen_kwargs)
            
            # Convert to ComfyUI format
            pil_image = images[0]
            comfy_image = pil_to_comfy(pil_image)
            
            return (comfy_image,)
            
        except Exception as e:
            print(f"Error in text to image generation: {str(e)}")
            raise e


class OvisU1ImageEdit:
    """ComfyUI node for editing images using text prompts with Ovis-U1."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("OVIS_MODEL",),
                "image": ("IMAGE",),
                "prompt": ("STRING", {"multiline": True, "default": "add a hat to this cat"}),
                "steps": ("INT", {"default": 50, "min": 1, "max": 200}),
                "txt_cfg": ("FLOAT", {"default": 6.0, "min": 0.0, "max": 20.0, "step": 0.1}),
                "img_cfg": ("FLOAT", {"default": 1.5, "min": 0.0, "max": 20.0, "step": 0.1}),
                "seed": ("INT", {"default": 42}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "edit_image"
    CATEGORY = "Ovis-U1"
    DESCRIPTION = "Edit image using text prompt with Ovis-U1"
    
    def edit_image(self, model, image, prompt, steps, txt_cfg, img_cfg, seed):
        """Edit an image based on text prompt using Ovis-U1 model.
        
        Args:
            model: OvisModelWrapper instance
            image: ComfyUI image tensor to edit
            prompt: Text description of desired edits
            steps: Number of diffusion steps
            txt_cfg: Text guidance scale
            img_cfg: Image guidance scale
            seed: Random seed for generation
            
        Returns:
            tuple: (ComfyUI image tensor,) containing the edited image
        """
        try:
            # Convert input image format
            input_img = comfy_to_pil(image)
            
            ovis_model = model.model
            text_tokenizer = model.text_tokenizer
            visual_tokenizer = model.visual_tokenizer
            
            # Smart resize
            width, height = input_img.size
            height, width = visual_tokenizer.smart_resize(height, width, factor=32)
            
            gen_kwargs = dict(
                max_new_tokens=1024,
                do_sample=False,
                top_p=None,
                top_k=None,
                temperature=None,
                repetition_penalty=None,
                eos_token_id=text_tokenizer.eos_token_id,
                pad_token_id=text_tokenizer.pad_token_id,
                use_cache=True,
                height=height,
                width=width,
                num_steps=steps,
                seed=seed,
                img_cfg=img_cfg,
                txt_cfg=txt_cfg,
            )
            
            # Generate unconditional baseline
            uncond_image = load_blank_image(width, height)
            uncond_prompt = "<image>\nGenerate an image."
            input_ids, pixel_values, attention_mask, grid_thws, _ = build_inputs(
                ovis_model, text_tokenizer, visual_tokenizer, uncond_prompt, uncond_image, width, height)
            
            with torch.inference_mode():
                no_both_cond = ovis_model.generate_condition(
                    input_ids, pixel_values=pixel_values, attention_mask=attention_mask, 
                    grid_thws=grid_thws, **gen_kwargs)
            
            # Generate no-text condition
            input_img_resized = input_img.resize((width, height))
            with torch.inference_mode():
                input_ids, pixel_values, attention_mask, grid_thws, _ = build_inputs(
                    ovis_model, text_tokenizer, visual_tokenizer, uncond_prompt, input_img_resized, width, height)
                no_txt_cond = ovis_model.generate_condition(
                    input_ids, pixel_values=pixel_values, attention_mask=attention_mask, 
                    grid_thws=grid_thws, **gen_kwargs)
            
            # Generate full condition
            prompt = "<image>\n" + prompt.strip()
            input_ids, pixel_values, attention_mask, grid_thws, vae_pixel_values = build_inputs(
                ovis_model, text_tokenizer, visual_tokenizer, prompt, input_img_resized, width, height)
            
            with torch.inference_mode():
                cond = ovis_model.generate_condition(
                    input_ids, pixel_values=pixel_values, attention_mask=attention_mask, 
                    grid_thws=grid_thws, **gen_kwargs)
                cond["vae_pixel_values"] = vae_pixel_values
                images = ovis_model.generate_img(
                    cond=cond, no_both_cond=no_both_cond, no_txt_cond=no_txt_cond, **gen_kwargs)
            
            # Convert to ComfyUI format
            pil_image = images[0]
            comfy_image = pil_to_comfy(pil_image)
            
            return (comfy_image,)
            
        except Exception as e:
            print(f"Error in image editing: {str(e)}")
            raise e


class OvisU1ImageToText:
    """ComfyUI node for generating text descriptions from images using Ovis-U1."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("OVIS_MODEL",),
                "image": ("IMAGE",),
                "prompt": ("STRING", {"default": "What is it?", "multiline": True}),
                "max_new_tokens": ("INT", {"default": 4096, "min": 1, "max": 8192}),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    FUNCTION = "image_to_text"
    CATEGORY = "Ovis-U1"
    DESCRIPTION = "Generate text description from image using Ovis-U1"
    
    def image_to_text(self, model, image, prompt, max_new_tokens):
        """Generate text description from image using Ovis-U1 model.
        
        Args:
            model: OvisModelWrapper instance
            image: ComfyUI image tensor to analyze
            prompt: Text prompt for the description task
            max_new_tokens: Maximum number of tokens to generate
            
        Returns:
            tuple: (Generated text,)
        """
        try:
            # Convert input image format
            pil_image = comfy_to_pil(image)
            
            ovis_model = model.model
            text_tokenizer = model.text_tokenizer
            visual_tokenizer = model.visual_tokenizer
            
            gen_kwargs = dict(
                max_new_tokens=max_new_tokens,
                do_sample=False,
                top_p=None,
                top_k=None,
                temperature=None,
                repetition_penalty=None,
                eos_token_id=text_tokenizer.eos_token_id,
                pad_token_id=text_tokenizer.pad_token_id,
                use_cache=True,
            )
            
            # Build inputs
            prompt = "<image>\n" + prompt
            input_ids, pixel_values, attention_mask, grid_thws = build_inputs(
                ovis_model, text_tokenizer, visual_tokenizer, prompt, pil_image)[:4]
            
            # Generate text
            with torch.inference_mode():
                output_ids = ovis_model.generate(
                    input_ids, pixel_values=pixel_values, attention_mask=attention_mask, 
                    grid_thws=grid_thws, **gen_kwargs)[0]
                gen_text = text_tokenizer.decode(output_ids, skip_special_tokens=True)
            
            return (gen_text,)
            
        except Exception as e:
            print(f"Error in image to text generation: {str(e)}")
            raise e


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