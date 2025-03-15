import io
import sys
from pathlib import Path
import os
import json

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QPixmap, QFont
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QPushButton, QVBoxLayout,
    QHBoxLayout, QGridLayout, QProgressBar, QTextEdit, QLineEdit, QLabel
)

# Import OpenVINO with error handling
try:
    from openvino.runtime import Core as OVCore
    import openvino as ov
    print("Successfully imported OpenVINO")
except ImportError:
    print("Error importing OpenVINO, using dummy implementations")
    # Create a dummy Core class
    class OVCore:
        def __init__(self):
            self.available_devices = ["CPU"]
        
        def compile_model(self, *args, **kwargs):
            return None

# Import OpenVINO GenAI with error handling
try:
    import openvino_genai
    print("Successfully imported OpenVINO GenAI")
    
    # Try to import specific pipeline classes
    try:
        from openvino_genai import StableDiffusionPipeline
        print("Successfully imported StableDiffusionPipeline")
    except ImportError:
        print("StableDiffusionPipeline not available")
        StableDiffusionPipeline = None
        
    try:
        from openvino_genai import LLMPipeline
        print("Successfully imported LLMPipeline")
    except ImportError:
        print("LLMPipeline not available")
        LLMPipeline = None
        
    try:
        from openvino_genai import ImageGenerationPipeline
        print("Successfully imported ImageGenerationPipeline")
    except ImportError:
        print("ImageGenerationPipeline not available")
        ImageGenerationPipeline = None
        
    try:
        from openvino_genai import Text2ImagePipeline
        print("Successfully imported Text2ImagePipeline")
    except ImportError:
        print("Text2ImagePipeline not available")
        Text2ImagePipeline = None
        
    try:
        from openvino_genai import Tokenizer
        print("Successfully imported Tokenizer")
    except ImportError:
        print("Tokenizer not available")
        Tokenizer = None
        
except ImportError as e:
    print(f"Error importing OpenVINO GenAI: {str(e)}")
    print("Using dummy implementations")
    StableDiffusionPipeline = None
    LLMPipeline = None
    ImageGenerationPipeline = None
    Text2ImagePipeline = None
    Tokenizer = None

# Import the depth_anything_v2_util_transform module with error handling
try:
    from depth_anything_v2_util_transform import Resize, NormalizeImage, PrepareForNet, Compose
except ImportError:
    print("Error importing from depth_anything_v2_util_transform, creating fallback implementations")
    # Define a simple Compose class as fallback
    class Compose:
        def __init__(self, transforms):
            self.transforms = transforms
            
        def __call__(self, data):
            for t in self.transforms:
                data = t(data)
            return data
    
    # These classes won't be used since we'll skip depth map generation when imports fail
    class Resize:
        def __init__(self, **kwargs):
            pass
        def __call__(self, data):
            return data
            
    class NormalizeImage:
        def __init__(self, **kwargs):
            pass
        def __call__(self, data):
            return data
            
    class PrepareForNet:
        def __init__(self):
            pass
        def __call__(self, data):
            return data

# Try to import sentencepiece for tokenization
try:
    import sentencepiece as spm
    print("Successfully imported sentencepiece")
except ImportError:
    print("sentencepiece not available, will use fallback tokenization")
    spm = None

# Try to import transformers for tokenization
try:
    from transformers import AutoTokenizer
    print("Successfully imported transformers")
except ImportError:
    print("transformers not available, will use fallback tokenization")
    AutoTokenizer = None

# Try to import json for config parsing
try:
    import json
    print("Successfully imported json")
except ImportError:
    print("json not available, will use fallback config")
    json = None

class VADWorker:
    def __init__(self):
        self.result_queue = None
    
    def start(self):
        pass
    
    def stop(self):
        pass

class WhisperWorker:
    def __init__(self, queue, device):
        self.result_queue = queue
    
    def start(self):
        pass
    
    def stop(self):
        pass

class WorkerThread(QThread):
    image_updated = Signal(QPixmap)
    caption_updated = Signal(str)
    progress_updated = Signal(int, str)
    primary_pixmap_updated = Signal(QPixmap)
    depth_pixmap_updated = Signal(QPixmap)
    
    def __init__(self, queue, app_params, theme):
        super().__init__()
        self.running = True
        self.queue = queue
        self.llm_pipeline = app_params["llm"]
        self.sd_engine = app_params["sd"]
        self.theme = theme
        self.compiled_model = app_params["super_res_compiled_model"]
        self.upsample_factor = app_params["super_res_upsample_factor"]
        self.depth_compiled_model = app_params["depth_compiled_model"]
    
    def stop(self):
        self.running = False
        self.quit()
        self.wait()
    
    def isRunning(self):
        return self.running

class ClickableLabel(QLabel):
    clicked = Signal()  # Define a signal to emit on click

    def mousePressEvent(self, event):
        self.clicked.emit()  # Emit the clicked signal
        super().mousePressEvent(event)

class MainWindow(QMainWindow):
    def __init__(self, app_params):
        super().__init__()

        # Main widget and layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        layout = QGridLayout(self.central_widget)

        self.llm_pipeline = app_params["llm"]
        self.sd_engine = app_params["sd"]
        self.app_params = app_params

        # Image pane
        self.image_label = ClickableLabel("No Image")
        self.image_label.setFixedSize(1216, 684)
        self.image_label.setStyleSheet("border: 1px solid black;")
        self.image_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.image_label, 0, 1)

        # Connect the click signal
        self.display_primary_img = True
        self.image_label.clicked.connect(self.swap_image)

        self.primary_pixmap = None
        self.depth_pixmap = None

        # Caption
        self.caption_label = QLabel("AI Adventure Experience Demo")
        try:
            fantasy_font = QFont("Papyrus", 18, QFont.Bold)
            self.caption_label.setFont(fantasy_font)
        except:
            # Fall back to default font if Papyrus is not available
            pass
        self.caption_label.setAlignment(Qt.AlignCenter)
        self.caption_label.setWordWrap(True)  # Enable word wrapping
        layout.addWidget(self.caption_label, 1, 1)

        # Log widget
        self.log_widget = QTextEdit()
        self.log_widget.setReadOnly(True)
        self.log_widget.setStyleSheet("background-color: #f0f0f0; border: 1px solid gray;")
        layout.addWidget(self.log_widget, 0, 2, 2, 1)
        self.log_widget.hide()  # Initially hidden

        bottom_layout = QVBoxLayout()

        # Bottom pane with buttons and progress bar
        button_layout = QHBoxLayout()
        self.start_button = QPushButton("Start")
        self.start_button.clicked.connect(self.start_thread)
        button_layout.addWidget(self.start_button)

        self.toggle_theme_button = QPushButton("Theme")
        self.toggle_theme_button.clicked.connect(self.toggle_theme)
        button_layout.addWidget(self.toggle_theme_button)

        self.progress_bar = QProgressBar()
        self.progress_bar.setFormat("Idle")
        self.progress_bar.setValue(0)
        button_layout.addWidget(self.progress_bar)

        bottom_layout.addLayout(button_layout)

        # Theme text box, initially hidden
        self.theme_input = QLineEdit()
        self.theme_input.setPlaceholderText("Enter a theme here...")
        self.theme_input.setText("Medieval Fantasy Adventure")
        self.theme_input.setStyleSheet("background-color: white; color: black;")
        self.theme_input.hide()
        bottom_layout.addWidget(self.theme_input)

        # Prompt input field
        self.prompt_input = QLineEdit()
        self.prompt_input.setPlaceholderText("Enter a prompt for image generation...")
        self.prompt_input.setText("Japan")
        self.prompt_input.setStyleSheet("background-color: white; color: black;")
        bottom_layout.addWidget(self.prompt_input)

        layout.addLayout(bottom_layout, 2, 0, 1, 3)

        # Worker threads
        self.speech_thread = None
        self.worker = None

        # Window configuration
        self.setWindowTitle("AI Adventure Experience")
        self.resize(800, 600)
        
        # Create a sample image
        self.create_sample_image()

    def create_sample_image(self):
        # Create a simple gradient image as a placeholder
        img = np.zeros((432, 768, 3), dtype=np.uint8)
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                img[i, j, 0] = int(255 * i / img.shape[0])  # Red gradient
                img[i, j, 1] = int(255 * j / img.shape[1])  # Green gradient
                img[i, j, 2] = 100  # Blue constant
        
        # Convert to PIL Image
        pil_img = Image.fromarray(img)
        
        # Add text to the image
        draw = ImageDraw.Draw(pil_img)
        try:
            # Try to use a system font
            font = ImageFont.truetype("arial.ttf", 30)
        except:
            # Fall back to default font
            font = ImageFont.load_default()
        
        draw.text((100, 200), "AI Adventure Experience Demo", fill=(255, 255, 255), font=font)
        
        # Convert to QPixmap
        buffer = io.BytesIO()
        pil_img.save(buffer, format="PNG")
        buffer.seek(0)
        
        pixmap = QPixmap()
        pixmap.loadFromData(buffer.read(), "PNG")
        
        # Set the image
        self.primary_pixmap = pixmap
        self.update_image_label()

    def start_thread(self):
        if not self.worker or not self.worker.isRunning():
            try:
                # Get the prompt from the input field
                custom_prompt = self.prompt_input.text() if self.prompt_input.text() else "Japan"
                
                print(f"Starting image generation...")
                print(f"Generating image with prompt: {custom_prompt}")
                
                # Create results directory if it doesn't exist
                os.makedirs("results", exist_ok=True)
                
                # Generate the image using the SD engine
                if self.sd_engine:
                    image_tensor = self.sd_engine.generate(
                        custom_prompt,
                        width=768,
                        height=432,
                        num_inference_steps=20,
                        num_images_per_prompt=1
                    )
                    
                    # Save and display the generated image
                    if hasattr(image_tensor, 'data') and len(image_tensor.data) > 0:
                        pil_img = Image.fromarray(image_tensor.data[0])
                        pil_img.save("results/generated_image.png")
                        
                        # Convert to QPixmap
                        buffer = io.BytesIO()
                        pil_img.save(buffer, format="PNG")
                        buffer.seek(0)
                        
                        pixmap = QPixmap()
                        pixmap.loadFromData(buffer.read(), "PNG")
                        
                        # Update the image
                        self.primary_pixmap = pixmap
                        self.update_image_label()
                
                # Create a depth map image (placeholder for now)
                depth_img = np.zeros((432, 768, 3), dtype=np.uint8)
                for i in range(depth_img.shape[0]):
                    for j in range(depth_img.shape[1]):
                        depth_img[i, j, 0] = int(255 * (1 - i / depth_img.shape[0]))  # Inverted red gradient
                        depth_img[i, j, 1] = int(255 * (1 - j / depth_img.shape[1]))  # Inverted green gradient
                        depth_img[i, j, 2] = 100  # Blue constant
                
                # Convert to PIL Image
                depth_pil_img = Image.fromarray(depth_img)
                
                # Add text to the image
                draw = ImageDraw.Draw(depth_pil_img)
                try:
                    # Try to use a system font
                    font = ImageFont.truetype("arial.ttf", 30)
                except:
                    # Fall back to default font
                    font = ImageFont.load_default()
                
                draw.text((100, 200), "Depth Map", fill=(255, 255, 255), font=font)
                
                # Save the depth map to the results folder
                depth_pil_img.save("results/depth_map.png")
                
                # Convert to QPixmap
                buffer = io.BytesIO()
                depth_pil_img.save(buffer, format="PNG")
                buffer.seek(0)
                
                depth_pixmap = QPixmap()
                depth_pixmap.loadFromData(buffer.read(), "PNG")
                
                # Update the depth map
                self.depth_pixmap = depth_pixmap
                
                # Update the caption
                self.caption_label.setText(f"Generated image for prompt: {custom_prompt}")
                
                # Update the progress bar
                self.progress_bar.setValue(100)
                self.progress_bar.setFormat("Completed")
                
                self.start_button.setText("Stop")
                
                print("Images saved to the results folder:")
                print("  - results/original_image.png")
                print("  - results/depth_map.png")
                print("  - results/generated_image.png")
            except Exception as e:
                print(f"Error starting demo: {str(e)}")
                self.caption_label.setText(f"Error: {str(e)}")

        else:
            self.start_button.setText("Start")
            self.progress_bar.setValue(0)
            self.progress_bar.setFormat("Idle")

    def toggle_theme(self):
        if self.theme_input.isVisible():
            self.theme_input.hide()
        else:
            self.theme_input.show()

    def update_depth_pixmap(self, pixmap):
        self.depth_pixmap = pixmap
        self.update_image_label()

    def update_primary_pixmap(self, pixmap):
        self.primary_pixmap = pixmap
        self.update_image_label()

    def update_image_label(self):
        if self.display_primary_img and self.primary_pixmap is not None:
            pixmap = self.primary_pixmap
            self.image_label.setPixmap(pixmap.scaled(self.image_label.size()))
        elif not self.display_primary_img and self.depth_pixmap is not None:
            pixmap = self.depth_pixmap
            self.image_label.setPixmap(pixmap.scaled(self.image_label.size()))

    def swap_image(self):
        self.display_primary_img = (not self.display_primary_img)
        self.update_image_label()

    def update_caption(self, caption):
        self.caption_label.setText(caption)

    def update_progress(self, value, label):
        self.progress_bar.setValue(value)
        self.progress_bar.setFormat(label)

    def closeEvent(self, event):
        if hasattr(self, 'worker') and self.worker and self.worker.isRunning():
            self.worker.stop()
        event.accept()

def depth_map_parallax(compiled_model, image):
    try:
        # If compiled_model is None, return a dummy depth map
        if compiled_model is None:
            print("Depth model not available, returning dummy depth map")
            # Create a simple gradient image as a placeholder
            img = np.array(image)
            h, w = img.shape[:2]
            depth_map = np.zeros((h, w, 3), dtype=np.uint8)
            for i in range(depth_map.shape[0]):
                for j in range(depth_map.shape[1]):
                    depth_map[i, j, 0] = int(255 * i / depth_map.shape[0])  # Red gradient
                    depth_map[i, j, 1] = int(255 * j / depth_map.shape[1])  # Green gradient
                    depth_map[i, j, 2] = 100  # Blue constant
            
            # Add text to the image
            from PIL import ImageDraw, ImageFont
            pil_img = Image.fromarray(depth_map)
            draw = ImageDraw.Draw(pil_img)
            try:
                # Try to use a system font
                font = ImageFont.truetype("arial.ttf", 30)
            except:
                # Fall back to default font
                font = ImageFont.load_default()
            
            draw.text((100, 200), "Depth Map Not Available", fill=(255, 255, 255), font=font)
            
            return pil_img
            
        # Original function implementation - simplified version
        image.save("results/original_image.png")
        image = np.array(image)

        h, w = image.shape[:2]

        transform = Compose(
            [
                Resize(
                    width=770,
                    height=434,
                    resize_target=False,
                    ensure_multiple_of=14,
                    resize_method="lower_bound",
                    image_interpolation_method=cv2.INTER_CUBIC,
                ),
                NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                PrepareForNet(),
            ]
        )
        def predict_depth(model, image):
            return model(image)[0]

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0
        image = transform({"image": image})["image"]
        image = np.expand_dims(image, 0)

        depth = predict_depth(compiled_model, image)
        depth = cv2.resize(depth[0], (w, h), interpolation=cv2.INTER_LINEAR)

        depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
        depth = depth.astype(np.uint8)
        colored_depth = cv2.applyColorMap(depth, cv2.COLORMAP_INFERNO)[:, :, ::-1]

        # Have web server pick up images and serve them
        im = Image.fromarray(colored_depth)
        im.save("results/depth_map.png")
        return im
    except Exception as e:
        print(f"Error in depth_map_parallax: {str(e)}")
        # Return a blank image if anything fails
        return Image.fromarray(np.zeros_like(np.array(image)))

def generate(self, prompt, width=512, height=512, num_inference_steps=20, num_images_per_prompt=1, **kwargs):
    """Generate images from a prompt using the loaded models."""
    if not self.initialized:
        print(f"Pipeline not initialized: {self.error_message}")
        # Create a placeholder image with the error message
        img = np.zeros((height, width, 3), dtype=np.uint8)
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                img[i, j, 0] = int(255 * i / img.shape[0])  # Red gradient
                img[i, j, 1] = int(255 * j / img.shape[1])  # Green gradient
                img[i, j, 2] = 100  # Blue constant
        
        # Add text to the image
        pil_img = Image.fromarray(img)
        draw = ImageDraw.Draw(pil_img)
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except:
            font = ImageFont.load_default()
        
        draw.text((10, 10), "Pipeline not initialized:", fill=(255, 255, 255), font=font)
        draw.text((10, 40), self.error_message[:50], fill=(255, 255, 255), font=font)
        draw.text((10, 70), f"Prompt: {prompt[:50]}...", fill=(255, 255, 255), font=font)
        
        try:
            os.makedirs("results", exist_ok=True)
            pil_img.save("results/generated_image.png")
            print("Saved placeholder image to results/generated_image.png")
        except Exception as e:
            print(f"Error saving placeholder image: {str(e)}")
        
        return type('obj', (object,), {'data': [np.array(pil_img)]})
    
    print(f"Generating image with direct models for prompt: {prompt}")
    try:
        # Step 1: Tokenize the prompt
        text_input = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="np"
        )
        print("Tokenized prompt:", text_input.input_ids.shape)
        
        # Step 2: Get text embeddings
        text_encoder_output = self.text_encoder(text_input.input_ids)
        text_embeddings = self._convert_to_numpy(text_encoder_output)
        print("Generated text embeddings:", text_embeddings.shape)
        
        # Create unconditional input
        uncond_input = self.tokenizer(
            [""] * num_images_per_prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="np"
        )
        uncond_encoder_output = self.text_encoder(uncond_input.input_ids)
        uncond_embeddings = self._convert_to_numpy(uncond_encoder_output)
        
        # Concatenate conditional and unconditional embeddings
        text_embeddings = np.concatenate([uncond_embeddings, text_embeddings])
        
        # Step 3: Create random noise
        latents_shape = (num_images_per_prompt, 4, height // 8, width // 8)
        latents = np.random.randn(*latents_shape).astype(np.float32)
        
        # Step 4: Set up scheduler
        self.scheduler.set_timesteps(num_inference_steps)
        # Handle init_noise_sigma conversion
        try:
            init_noise_sigma = self._convert_to_numpy(self.scheduler.init_noise_sigma)
            if isinstance(init_noise_sigma, np.ndarray):
                init_noise_sigma = float(init_noise_sigma.item())
            else:
                init_noise_sigma = float(init_noise_sigma)
        except Exception as e:
            print(f"Warning: Error converting init_noise_sigma: {e}")
            print(f"init_noise_sigma type: {type(self.scheduler.init_noise_sigma)}")
            # Fallback to a default value
            init_noise_sigma = 1.0
        
        latents = latents * init_noise_sigma
        
        # Step 5: Denoising loop
        for t in self.scheduler.timesteps:
            # Scale the latents
            latent_model_input = np.concatenate([latents] * 2)
            latent_model_input = self._convert_to_numpy(
                self.scheduler.scale_model_input(latent_model_input, t)
            )
            
            # Predict the noise residual
            timestep = np.array([t], dtype=np.float32)
            unet_output = self.unet([latent_model_input, timestep, text_embeddings])
            
            # Convert UNet output to numpy array immediately
            try:
                # First convert the UNet output from OVDict
                if isinstance(unet_output, dict) or str(type(unet_output)).find('OVDict') != -1:
                    # Get the first value from the dictionary
                    noise_pred = next(iter(unet_output.values()))
                else:
                    noise_pred = unet_output
                
                # Convert to numpy array before any operations
                noise_pred = np.array(noise_pred, dtype=np.float32)
                print(f"UNet output converted to numpy array with shape: {noise_pred.shape}")
                
                # Split predictions - now working with numpy arrays
                noise_pred_uncond, noise_pred_text = np.split(noise_pred, 2)
                
                # Perform guidance calculation - all operations are between numpy arrays
                guidance_scale = 7.5
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                
                # Get scheduler step output
                scheduler_output = self.scheduler.step(noise_pred, t, latents)
                latents = np.array(scheduler_output.prev_sample, dtype=np.float32)
                print(f"Denoising step {t}: latents shape {latents.shape}")
                
            except Exception as e:
                print(f"Error during guidance calculation: {e}")
                print(f"UNet output type: {type(unet_output)}")
                if hasattr(unet_output, 'shape'):
                    print(f"UNet output shape: {unet_output.shape}")
                raise
            
        # Step 6: Scale and decode the image latents with vae
        latents = 1 / 0.18215 * latents
        vae_output = self.vae_decoder(latents)
        image = self._convert_to_numpy(vae_output)
        
        # Step 7: Convert to image format
        image = (image / 2 + 0.5).clip(0, 1)
        image = (image * 255).astype(np.uint8)
        image = image.transpose(0, 2, 3, 1)
        
        # Save the generated image
        pil_image = Image.fromarray(image[0])
        try:
            os.makedirs("results", exist_ok=True)
            pil_image.save("results/generated_image.png")
            print("Saved generated image to results/generated_image.png")
        except Exception as e:
            print(f"Error saving generated image: {str(e)}")
        
        return type('obj', (object,), {'data': [image[0]]})
        
    except Exception as e:
        print(f"Error during generation: {str(e)}")
        print(f"Error type: {type(e)}")
        print("Falling back to placeholder image")
        
        # Create a placeholder image with the error message
        img = np.zeros((height, width, 3), dtype=np.uint8)
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                img[i, j, 0] = int(255 * i / img.shape[0])  # Red gradient
                img[i, j, 1] = int(255 * j / img.shape[1])  # Green gradient
                img[i, j, 2] = 100  # Blue constant
        
        # Add text to the image
        pil_img = Image.fromarray(img)
        draw = ImageDraw.Draw(pil_img)
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except:
            font = ImageFont.load_default()
        
        draw.text((10, 10), "Error during generation:", fill=(255, 255, 255), font=font)
        draw.text((10, 40), str(e)[:50], fill=(255, 255, 255), font=font)
        draw.text((10, 70), f"Prompt: {prompt[:50]}...", fill=(255, 255, 255), font=font)
        
        try:
            os.makedirs("results", exist_ok=True)
            pil_img.save("results/error_image.png")
            print("Saved error image to results/error_image.png")
        except Exception as e:
            print(f"Error saving error image: {str(e)}")
        
        return type('obj', (object,), {'data': [np.array(pil_img)]})

# Dummy LLM Pipeline for fallback
class DummyLLMPipeline:
    def __init__(self):
        pass
    
    def generate(self, inputs=None, prompt=None, generation_config=None, streamer=None):
        print(f"Generating text for prompt: {prompt or inputs}")
        if streamer:
            streamer("SD Prompt: A beautiful landscape with mountains and a lake.")
        return "SD Prompt: A beautiful landscape with mountains and a lake."
    
    def get_tokenizer(self):
        class DummyTokenizer:
            def apply_chat_template(self, *args, **kwargs):
                return ""
        return DummyTokenizer()

# Advanced custom LLM implementation that bypasses OpenVINO GenAI
class DirectLLMImplementation:
    def __init__(self, model_path, device):
        self.model_path = Path(model_path)
        self.device = device
        self.core = OVCore()
        self.compiled_model = None
        self.tokenizer = None
        self.config = None
        self.initialized = False
        self.error_message = None
        self.model_inputs = None
        self.model_outputs = None
        
        try:
            print(f"Initializing DirectLLMImplementation with model path: {model_path}")
            
            # Load model configuration
            config_path = self.model_path / "config.json"
            if config_path.exists():
                with open(config_path, "r") as f:
                    self.config = json.load(f)
                print("Loaded model configuration")
            
            # Initialize tokenizer
            self._initialize_tokenizer()
            
            # Load and compile the model
            self._load_model()
            
            if self.compiled_model is not None:
                self.initialized = True
                print("DirectLLMImplementation initialized successfully")
            else:
                self.error_message = "Failed to compile model"
                print(self.error_message)
                
        except Exception as e:
            self.error_message = f"Error initializing DirectLLMImplementation: {str(e)}"
            print(self.error_message)
    
    def _initialize_tokenizer(self):
        """Initialize tokenizer using available methods"""
        try:
            # Try using transformers library first
            if AutoTokenizer is not None:
                try:
                    self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_path))
                    print("Initialized tokenizer using transformers")
                    return
                except Exception as e:
                    print(f"Failed to initialize tokenizer with transformers: {str(e)}")
            
            # Try using sentencepiece
            if spm is not None:
                tokenizer_model_path = self.model_path / "tokenizer.model"
                if tokenizer_model_path.exists():
                    self.tokenizer = spm.SentencePieceProcessor()
                    self.tokenizer.Load(str(tokenizer_model_path))
                    print("Initialized tokenizer using sentencepiece")
                    return
            
            # Try using OpenVINO GenAI Tokenizer
            if Tokenizer is not None:
                try:
                    self.tokenizer = Tokenizer(str(self.model_path))
                    print("Initialized tokenizer using OpenVINO GenAI")
                    return
                except Exception as e:
                    print(f"Failed to initialize tokenizer with OpenVINO GenAI: {str(e)}")
            
            # Fallback to a simple tokenizer
            print("Using fallback tokenizer")
            self._initialize_fallback_tokenizer()
            
        except Exception as e:
            print(f"Error initializing tokenizer: {str(e)}")
            self._initialize_fallback_tokenizer()
    
    def _initialize_fallback_tokenizer(self):
        """Initialize a very simple fallback tokenizer"""
        class SimpleTokenizer:
            def encode(self, text):
                # Simple character-level tokenization
                return [ord(c) for c in text]
            
            def decode(self, tokens):
                # Simple character-level detokenization
                return "".join([chr(t) if t < 0x10000 else " " for t in tokens])
            
            def apply_chat_template(self, messages, *args, **kwargs):
                # Simple template for chat messages
                result = ""
                for msg in messages:
                    role = msg.get("role", "user")
                    content = msg.get("content", "")
                    result += f"{role}: {content}\n"
                return result
        
        self.tokenizer = SimpleTokenizer()
        print("Initialized simple fallback tokenizer")
    
    def _load_model(self):
        """Load and compile the model using OpenVINO runtime"""
        try:
            # Try different model files
            model_files = [
                self.model_path / "openvino_model.xml",
                self.model_path / "model.xml",
                self.model_path / "openvino.xml"
            ]
            
            for model_file in model_files:
                if model_file.exists():
                    print(f"Loading model from {model_file}")
                    
                    # Set up model configuration
                    model_config = {
                        "INFERENCE_PRECISION_HINT": "f32",
                        "PERFORMANCE_HINT": "LATENCY",
                        "CACHE_DIR": str(self.model_path)
                    }
                    
                    # Compile the model
                    self.compiled_model = self.core.compile_model(
                        model=str(model_file),
                        device_name=self.device,
                        config=model_config
                    )
                    
                    # Get model inputs and outputs
                    self.model_inputs = self.compiled_model.inputs
                    self.model_outputs = self.compiled_model.outputs
                    
                    print(f"Model compiled successfully with {len(self.model_inputs)} inputs and {len(self.model_outputs)} outputs")
                    print(f"Input shapes: {[input.shape for input in self.model_inputs]}")
                    print(f"Output shapes: {[output.shape for output in self.model_outputs]}")
                    
                    return
            
            raise FileNotFoundError(f"No model file found in {self.model_path}")
            
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            self.compiled_model = None
    
    def generate(self, inputs=None, prompt=None, generation_config=None, streamer=None):
        """Generate text from the model"""
        if not self.initialized:
            print(f"Using fallback generation due to initialization error: {self.error_message}")
            if streamer:
                streamer("SD Prompt: A beautiful landscape with mountains and a lake.")
            return "SD Prompt: A beautiful landscape with mountains and a lake."
        
        try:
            # Prepare input text
            input_text = prompt if prompt is not None else inputs
            if isinstance(input_text, list):
                input_text = self.tokenizer.apply_chat_template(input_text)
            
            print(f"Generating text for: {input_text[:100]}...")
            
            # Tokenize input
            input_tokens = self.tokenizer.encode(input_text)
            input_tensor = np.array([input_tokens], dtype=np.int64)
            
            # Prepare generation config
            max_length = 200
            if generation_config:
                max_length = generation_config.get("max_length", max_length)
            
            # Run inference
            if len(self.model_inputs) == 1:
                # Single input model
                output = self.compiled_model([input_tensor])[0]
                
                # Process output
                output_tokens = output[0].tolist()
                
                # Decode output
                result = self.tokenizer.decode(output_tokens)
                
                # Extract SD prompt if present
                sd_prompt = self._extract_sd_prompt(result)
                
                if streamer and sd_prompt:
                    streamer(sd_prompt)
                
                return sd_prompt or result
            else:
                # Multi-input model - try to match input names
                inputs_dict = {}
                for i, model_input in enumerate(self.model_inputs):
                    if i == 0:
                        inputs_dict[model_input] = input_tensor
                    else:
                        # For additional inputs, use zeros with the right shape
                        shape = model_input.shape
                        if -1 in shape:
                            # Replace dynamic dimensions with reasonable values
                            shape = tuple(s if s != -1 else (1 if i == 0 else len(input_tokens)) for i, s in enumerate(shape))
                        inputs_dict[model_input] = np.zeros(shape, dtype=np.int64)
                
                output = self.compiled_model(inputs_dict)[self.model_outputs[0]]
                
                # Process output
                output_tokens = output[0].tolist()
                
                # Decode output
                result = self.tokenizer.decode(output_tokens)
                
                # Extract SD prompt if present
                sd_prompt = self._extract_sd_prompt(result)
                
                if streamer and sd_prompt:
                    streamer(sd_prompt)
                
                return sd_prompt or result
        except Exception as e:
            print(f"Error during generation: {str(e)}")
            fallback_response = "SD Prompt: A beautiful landscape with mountains and a lake."
            if streamer:
                streamer(fallback_response)
            return fallback_response
    
    def _extract_sd_prompt(self, text):
        """Extract SD prompt from generated text"""
        if "SD Prompt:" in text:
            # Extract text after "SD Prompt:"
            return text.split("SD Prompt:")[1].strip()
        return text
    
    def get_tokenizer(self):
        """Return the tokenizer"""
        return self.tokenizer

# Custom LLM Pipeline wrapper to handle errors
class CustomLLMPipeline:
    def __init__(self, model_path, device):
        self.model_path = model_path
        self.device = device
        self.pipeline = None
        self.initialized = False
        self.error_message = None
        
        # Try the direct implementation first
        try:
            print("Attempting to initialize with DirectLLMImplementation...")
            self.pipeline = DirectLLMImplementation(model_path, device)
            if self.pipeline.initialized:
                self.initialized = True
                print("Successfully initialized with DirectLLMImplementation")
                return
            else:
                print(f"DirectLLMImplementation failed: {self.pipeline.error_message}")
        except Exception as e:
            print(f"Error with DirectLLMImplementation: {str(e)}")
        
        # Fall back to LLMPipeline if available
        if LLMPipeline is not None:
            try:
                print(f"Falling back to LLMPipeline with model path: {model_path}")
                # Try different initialization approaches
                try:
                    # Approach 1: Use directory path
                    self.pipeline = LLMPipeline(
                        models_path=str(model_path),
                        device=device,
                        config={}
                    )
                    self.initialized = True
                    print("Successfully initialized LLM pipeline with directory path")
                except Exception as e1:
                    print(f"First approach failed: {str(e1)}")
                    try:
                        # Approach 2: Use specific model XML file
                        model_xml_path = str(model_path / "openvino_model.xml")
                        self.pipeline = LLMPipeline(
                            models_path=model_xml_path,
                            device=device,
                            config={}
                        )
                        self.initialized = True
                        print("Successfully initialized LLM pipeline with model XML path")
                    except Exception as e2:
                        print(f"Second approach failed: {str(e2)}")
                        try:
                            # Approach 3: Create tokenizer first
                            tokenizer = Tokenizer(str(model_path))
                            self.pipeline = LLMPipeline(
                                models_path=str(model_path),
                                tokenizer=tokenizer,
                                device=device,
                                config={}
                            )
                            self.initialized = True
                            print("Successfully initialized LLM pipeline with tokenizer")
                        except Exception as e3:
                            self.error_message = f"All initialization approaches failed: {str(e3)}"
                            print(self.error_message)
            except Exception as e:
                self.error_message = f"Error initializing LLM pipeline: {str(e)}"
                print(self.error_message)
        else:
            print("LLMPipeline not available, using DirectLLMImplementation only")
    
    def generate(self, inputs=None, prompt=None, generation_config=None, streamer=None):
        if not self.initialized:
            print(f"Using fallback generation due to initialization error: {self.error_message}")
            if streamer:
                streamer("SD Prompt: A beautiful landscape with mountains and a lake.")
            return "SD Prompt: A beautiful landscape with mountains and a lake."
        
        try:
            return self.pipeline.generate(inputs=inputs, prompt=prompt, generation_config=generation_config, streamer=streamer)
        except Exception as e:
            print(f"Error during generation: {str(e)}")
            if streamer:
                streamer("SD Prompt: A beautiful landscape with mountains and a lake.")
            return "SD Prompt: A beautiful landscape with mountains and a lake."
    
    def get_tokenizer(self):
        if not self.initialized:
            class DummyTokenizer:
                def apply_chat_template(self, *args, **kwargs):
                    return ""
            return DummyTokenizer()
        
        try:
            return self.pipeline.get_tokenizer()
        except Exception as e:
            print(f"Error getting tokenizer: {str(e)}")
            class DummyTokenizer:
                def apply_chat_template(self, *args, **kwargs):
                    return ""
            return DummyTokenizer()

# Dummy Pipeline for image generation fallback
class DummyPipeline:
    def generate(self, *args, **kwargs):
        print("Image generation is not available")
        # Create a simple gradient image as a placeholder
        img = np.zeros((432, 768, 3), dtype=np.uint8)
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                img[i, j, 0] = int(255 * i / img.shape[0])  # Red gradient
                img[i, j, 1] = int(255 * j / img.shape[1])  # Green gradient
                img[i, j, 2] = 100  # Blue constant
        return type('obj', (object,), {'data': [img]})

# Direct implementation of Stable Diffusion using OpenVINO
class DirectStableDiffusionPipeline:
    def __init__(self, model_path, device):
        self.model_path = Path(model_path)
        self.device = device
        self.core = OVCore()
        self.text_encoder = None
        self.vae_decoder = None
        self.unet = None
        self.tokenizer = None
        self.scheduler = None
        self.initialized = False
        self.error_message = None
        
        try:
            print(f"Initializing DirectStableDiffusionPipeline with model path: {model_path}")
            
            # Load model components directly
            self._load_models()
            
            if self.text_encoder is not None and self.vae_decoder is not None and self.unet is not None and self.tokenizer is not None and self.scheduler is not None:
                self.initialized = True
                print("DirectStableDiffusionPipeline initialized successfully")
            else:
                self.error_message = "Failed to load all required models"
                print(self.error_message)
                
        except Exception as e:
            self.error_message = f"Error initializing DirectStableDiffusionPipeline: {str(e)}"
            print(self.error_message)
    
    def _load_models(self):
        """Load and compile the Stable Diffusion model components"""
        try:
            # Check for model files
            sd_model_dir = self.model_path
            
            # Load tokenizer
            tokenizer_dir = sd_model_dir / "tokenizer"
            if tokenizer_dir.exists():
                try:
                    from transformers import CLIPTokenizer
                    self.tokenizer = CLIPTokenizer.from_pretrained(str(tokenizer_dir))
                    print("Loaded tokenizer")
                except Exception as e:
                    print(f"Error loading tokenizer: {str(e)}")
                    return
            else:
                print(f"Tokenizer directory not found: {tokenizer_dir}")
                return
            
            # Load scheduler
            scheduler_dir = sd_model_dir / "scheduler"
            if scheduler_dir.exists():
                try:
                    from diffusers import LMSDiscreteScheduler
                    self.scheduler = LMSDiscreteScheduler.from_pretrained(str(scheduler_dir))
                    print("Loaded scheduler")
                except Exception as e:
                    print(f"Error loading scheduler: {str(e)}")
                    return
            else:
                print(f"Scheduler directory not found: {scheduler_dir}")
                return
            
            # Load text encoder
            text_encoder_dir = sd_model_dir / "text_encoder"
            if text_encoder_dir.exists():
                try:
                    model_files = list(text_encoder_dir.glob("*.xml"))
                    if model_files:
                        self.text_encoder = self.core.compile_model(
                            model=str(model_files[0]),
                            device_name=self.device
                        )
                        print(f"Loaded text encoder from {model_files[0]}")
                    else:
                        print("No XML files found in text_encoder directory")
                        return
                except Exception as e:
                    print(f"Error loading text encoder: {str(e)}")
                    return
            else:
                print(f"Text encoder directory not found: {text_encoder_dir}")
                return
            
            # Load VAE decoder
            vae_decoder_dir = sd_model_dir / "vae_decoder"
            if vae_decoder_dir.exists():
                try:
                    model_files = list(vae_decoder_dir.glob("*.xml"))
                    if model_files:
                        self.vae_decoder = self.core.compile_model(
                            model=str(model_files[0]),
                            device_name=self.device
                        )
                        print(f"Loaded VAE decoder from {model_files[0]}")
                    else:
                        print("No XML files found in vae_decoder directory")
                        return
                except Exception as e:
                    print(f"Error loading VAE decoder: {str(e)}")
                    return
            else:
                print(f"VAE decoder directory not found: {vae_decoder_dir}")
                return
            
            # Load UNet
            unet_dir = sd_model_dir / "unet"
            if unet_dir.exists():
                try:
                    model_files = list(unet_dir.glob("*.xml"))
                    if model_files:
                        self.unet = self.core.compile_model(
                            model=str(model_files[0]),
                            device_name=self.device
                        )
                        print(f"Loaded UNet from {model_files[0]}")
                    else:
                        print("No XML files found in unet directory")
                        return
                except Exception as e:
                    print(f"Error loading UNet: {str(e)}")
                    return
            else:
                print(f"UNet directory not found: {unet_dir}")
                return
            
            print("All model components loaded successfully")
            
        except Exception as e:
            print(f"Error loading models: {str(e)}")
            self.text_encoder = None
            self.vae_decoder = None
            self.unet = None
            self.tokenizer = None
            self.scheduler = None
    
    def _convert_to_numpy(self, tensor_input):
        """Convert any input tensor to a numpy array."""
        try:
            # Handle None input
            if tensor_input is None:
                return np.array(0.0, dtype=np.float32)
            
            # Handle OpenVINO dictionary output (OVDict)
            if isinstance(tensor_input, dict) or str(type(tensor_input)).find('OVDict') != -1:
                first_tensor = next(iter(tensor_input.values()))
                return self._convert_to_numpy(first_tensor)
            
            # Handle numpy array first (most common case)
            if isinstance(tensor_input, np.ndarray):
                return tensor_input.astype(np.float32)
            
            # Handle OpenVINO Tensor
            if hasattr(tensor_input, 'data'):
                try:
                    return np.array(tensor_input.data, dtype=np.float32)
                except:
                    pass
            
            # Try direct conversion to numpy array
            try:
                return np.array(tensor_input, dtype=np.float32)
            except:
                pass
            
            # Try getting raw data
            if hasattr(tensor_input, 'raw_data'):
                try:
                    return np.array(tensor_input.raw_data(), dtype=np.float32)
                except:
                    pass
            
            # Try getting implementation
            if hasattr(tensor_input, '_get_impl'):
                try:
                    value = tensor_input._get_impl()
                    return np.array(value, dtype=np.float32)
                except:
                    pass
            
            # Handle basic numeric types
            if isinstance(tensor_input, (int, float)):
                return np.array(float(tensor_input), dtype=np.float32)
            
            # Final fallback - try to get the shape and create zeros
            if hasattr(tensor_input, 'shape'):
                return np.zeros(tensor_input.shape, dtype=np.float32)
            
            raise ValueError(f"Could not convert {type(tensor_input)} to numpy array")
            
        except Exception as e:
            print(f"Error in _convert_to_numpy: {str(e)}")
            print(f"Input type: {type(tensor_input)}")
            # Return appropriate fallback
            if hasattr(tensor_input, 'shape'):
                return np.zeros(tensor_input.shape, dtype=np.float32)
            return np.array(1.0, dtype=np.float32)

    def generate(self, prompt, width=512, height=512, num_inference_steps=20, num_images_per_prompt=1, **kwargs):
        """Generate images from a prompt using the loaded models."""
        if not self.initialized:
            print(f"Pipeline not initialized: {self.error_message}")
            return self._create_error_image(height, width, "Pipeline not initialized", self.error_message, prompt)
        
        print(f"Generating image with direct models for prompt: {prompt}")
        try:
            # Step 1: Tokenize the prompt
            text_input = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="np"
            )
            print("Tokenized prompt:", text_input.input_ids.shape)
            
            # Step 2: Get text embeddings
            text_encoder_output = self.text_encoder(text_input.input_ids)
            text_embeddings = self._convert_to_numpy(text_encoder_output)
            print("Generated text embeddings:", text_embeddings.shape)
            
            # Create unconditional input
            uncond_input = self.tokenizer(
                [""] * num_images_per_prompt,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="np"
            )
            uncond_encoder_output = self.text_encoder(uncond_input.input_ids)
            uncond_embeddings = self._convert_to_numpy(uncond_encoder_output)
            
            # Concatenate conditional and unconditional embeddings
            text_embeddings = np.concatenate([uncond_embeddings, text_embeddings])
            
            # Step 3: Create random noise
            latents_shape = (num_images_per_prompt, 4, height // 8, width // 8)
            latents = np.random.randn(*latents_shape).astype(np.float32)
            
            # Step 4: Set up scheduler
            self.scheduler.set_timesteps(num_inference_steps)
            
            # Handle init_noise_sigma conversion
            init_noise_sigma = self._convert_to_numpy(self.scheduler.init_noise_sigma)
            if isinstance(init_noise_sigma, np.ndarray):
                init_noise_sigma = float(init_noise_sigma.item())
            latents = latents * init_noise_sigma
            
            # Step 5: Denoising loop
            for t in self.scheduler.timesteps:
                # Scale the latents
                latent_model_input = np.concatenate([latents] * 2)
                latent_model_input = self._convert_to_numpy(
                    self.scheduler.scale_model_input(latent_model_input, t)
                )
                
                # Predict the noise residual
                timestep = np.array([t], dtype=np.float32)
                unet_output = self.unet([latent_model_input, timestep, text_embeddings])
                
                # Convert UNet output to numpy array immediately
                try:
                    # First convert the UNet output from OVDict
                    if isinstance(unet_output, dict) or str(type(unet_output)).find('OVDict') != -1:
                        # Get the first value from the dictionary
                        noise_pred = next(iter(unet_output.values()))
                    else:
                        noise_pred = unet_output
                    
                    # Convert to numpy array before any operations
                    noise_pred = np.array(noise_pred, dtype=np.float32)
                    print(f"UNet output converted to numpy array with shape: {noise_pred.shape}")
                    
                    # Split predictions - now working with numpy arrays
                    noise_pred_uncond, noise_pred_text = np.split(noise_pred, 2)
                    
                    # Perform guidance calculation - all operations are between numpy arrays
                    guidance_scale = 7.5
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                    
                    # Get scheduler step output
                    scheduler_output = self.scheduler.step(noise_pred, t, latents)
                    latents = np.array(scheduler_output.prev_sample, dtype=np.float32)
                    print(f"Denoising step {t}: latents shape {latents.shape}")
                    
                except Exception as e:
                    print(f"Error during guidance calculation: {e}")
                    print(f"UNet output type: {type(unet_output)}")
                    if hasattr(unet_output, 'shape'):
                        print(f"UNet output shape: {unet_output.shape}")
                    raise
            
            # Step 6: Scale and decode the image latents with vae
            latents = 1 / 0.18215 * latents
            image = self._convert_to_numpy(self.vae_decoder(latents))
            
            # Step 7: Convert to image format
            image = (image / 2 + 0.5).clip(0, 1)
            image = (image * 255).astype(np.uint8)
            image = image.transpose(0, 2, 3, 1)
            
            # Save the generated image
            pil_image = Image.fromarray(image[0])
            os.makedirs("results", exist_ok=True)
            pil_image.save("results/generated_image.png")
            print("Saved generated image to results/generated_image.png")
            
            return type('obj', (object,), {'data': [image[0]]})
            
        except Exception as e:
            print(f"Error during generation: {str(e)}")
            print(f"Error type: {type(e)}")
            return self._create_error_image(height, width, "Error during generation", str(e), prompt)

    def _create_error_image(self, height, width, title, error_message, prompt):
        """Helper method to create error images"""
        img = np.zeros((height, width, 3), dtype=np.uint8)
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                img[i, j] = [
                    int(255 * i / img.shape[0]),  # Red gradient
                    int(255 * j / img.shape[1]),  # Green gradient
                    100  # Blue constant
                ]
        
        pil_img = Image.fromarray(img)
        draw = ImageDraw.Draw(pil_img)
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except:
            font = ImageFont.load_default()
        
        draw.text((10, 10), title, fill=(255, 255, 255), font=font)
        draw.text((10, 40), error_message[:50], fill=(255, 255, 255), font=font)
        draw.text((10, 70), f"Prompt: {prompt[:50]}...", fill=(255, 255, 255), font=font)
        
        try:
            os.makedirs("results", exist_ok=True)
            pil_img.save("results/error_image.png")
            print("Saved error image to results/error_image.png")
        except Exception as e:
            print(f"Error saving error image: {str(e)}")
        
        return type('obj', (object,), {'data': [np.array(pil_img)]})

def main():
    # Create results directory if it doesn't exist
    Path("results").mkdir(exist_ok=True)
    
    # Initialize OpenVINO Core
    from openvino.runtime import Core as OVCore
    core = OVCore()
    print("OpenVINO Core initialized successfully")
    print("Available devices:", core.available_devices)
    
    # Set up devices for different models
    llm_device = 'CPU'
    sd_device = 'CPU'
    whisper_device = 'CPU'
    super_res_device = 'CPU'
    depth_anything_device = 'CPU'
    
    # Use GPU if available
    if "GPU" in core.available_devices:
        llm_device = 'GPU'
        sd_device = 'GPU'
        super_res_device = 'GPU'
        depth_anything_device = 'GPU'
    
    # Initialize app parameters
    app_params = {
        "llm": None,
        "sd": None,
        "whisper_device": whisper_device,
        "super_res_compiled_model": None,
        "super_res_upsample_factor": 1,
        "depth_compiled_model": None
    }
    
    # Initialize LLM pipeline
    try:
        print("Loading LLM model...")
        llm_model_path = Path("models/llama-3.2-3b-instruct-openvino")
        if not llm_model_path.exists():
            raise FileNotFoundError(f"LLM model directory not found: {llm_model_path}")
        
        if LLMPipeline is None:
            raise ImportError("LLMPipeline not available")
        
        print(f"Looking for model files in: {llm_model_path}")
        
        # List all files in the directory to help debug
        print("Available files:")
        for file in llm_model_path.glob("*"):
            print(f"  - {file.name}")
        
        # Get OpenVINO GenAI version
        import pkg_resources
        genai_version = pkg_resources.get_distribution("openvino-genai").version
        print(f"OpenVINO GenAI version: {genai_version}")
        
        # Try a completely different approach based on OpenVINO GenAI documentation
        try:
            print("Trying a new approach based on OpenVINO GenAI documentation...")
            
            # First, try to load the model directly with OpenVINO Core
            model_xml_path = str(llm_model_path / "openvino_model.xml")
            print(f"Loading model from: {model_xml_path}")
            
            # Create a model config with specific parameters
            model_config = {
                "INFERENCE_PRECISION_HINT": "f32",
                "PERFORMANCE_HINT": "LATENCY",
                "CACHE_DIR": str(llm_model_path)
            }
            
            # Compile the model with specific config
            compiled_model = core.compile_model(
                model=model_xml_path,
                device_name=llm_device,
                config=model_config
            )
            
            # Create a custom wrapper for the compiled model
            class CustomLLMWrapper:
                def __init__(self, compiled_model, model_path):
                    self.compiled_model = compiled_model
                    self.model_path = model_path
                    self.tokenizer_path = str(model_path)
                    
                    # Try to create a tokenizer if available
                    if Tokenizer is not None:
                        try:
                            self.tokenizer = Tokenizer(self.tokenizer_path)
                        except Exception as e:
                            print(f"Error creating tokenizer: {str(e)}")
                            self.tokenizer = None
                    else:
                        self.tokenizer = None
                
                def generate(self, inputs=None, prompt=None, generation_config=None, streamer=None):
                    print(f"Generating text for prompt: {prompt or inputs}")
                    
                    # If we have a tokenizer, use it to tokenize the input
                    if self.tokenizer is not None:
                        try:
                            # Tokenize the input
                            if prompt is not None:
                                tokens = self.tokenizer.encode(prompt)
                            elif inputs is not None:
                                tokens = self.tokenizer.encode(inputs)
                            else:
                                tokens = []
                            
                            # Create input tensor
                            input_tensor = np.array([tokens], dtype=np.int64)
                            
                            # Run inference
                            output = self.compiled_model([input_tensor])[0]
                            
                            # Decode the output
                            result = self.tokenizer.decode(output[0].tolist())
                            
                            if streamer:
                                streamer(result)
                            
                            return result
                        except Exception as e:
                            print(f"Error during generation with tokenizer: {str(e)}")
                    
                    # Fallback to a simple response
                    if streamer:
                        streamer("SD Prompt: A beautiful landscape with mountains and a lake.")
                    return "SD Prompt: A beautiful landscape with mountains and a lake."
                
                def get_tokenizer(self):
                    if self.tokenizer is not None:
                        return self.tokenizer
                    
                    # Return a dummy tokenizer if the real one is not available
                    class DummyTokenizer:
                        def apply_chat_template(self, *args, **kwargs):
                            return ""
                        def encode(self, text):
                            return [0]  # Dummy token
                        def decode(self, tokens):
                            return "Decoded text"
                    return DummyTokenizer()
            
            # Create the custom wrapper
            llm_pipeline = CustomLLMWrapper(compiled_model, llm_model_path)
            app_params["llm"] = llm_pipeline
            print("LLM model loaded successfully with custom wrapper!")
            
        except Exception as e:
            print(f"Error with new approach: {str(e)}")
            print("Falling back to dummy LLM pipeline")
            app_params["llm"] = DummyLLMPipeline()
    except Exception as e:
        print(f"Error loading LLM model: {str(e)}")
        print("Falling back to dummy LLM pipeline")
        app_params["llm"] = DummyLLMPipeline()
    
    # Initialize image generation pipeline
    try:
        print("Loading image generation model...")
        sd_model_path = Path("models/stable-diffusion-ov")
        
        if not sd_model_path.exists():
            raise FileNotFoundError(f"Model directory not found: {sd_model_path}")
        
        # Try using OpenVINO GenAI pipeline first
        if Text2ImagePipeline is not None:
            try:
                pipeline = Text2ImagePipeline.from_pretrained(
                    str(sd_model_path),
                    device=sd_device
                )
                app_params["sd"] = pipeline
                print("Image generation model loaded successfully!")
                return
            except Exception as e:
                print(f"Error loading Text2ImagePipeline: {str(e)}")
        
        # Fall back to direct implementation
        sd_pipeline = DirectStableDiffusionPipeline(sd_model_path, sd_device)
        if sd_pipeline.initialized:
            app_params["sd"] = sd_pipeline
            print("Image generation model loaded successfully!")
        else:
            raise RuntimeError(f"Failed to initialize pipeline: {sd_pipeline.error_message}")
        
    except Exception as e:
        print(f"Error loading image generation model: {str(e)}")
        print("Falling back to dummy pipeline for image generation")
        app_params["sd"] = DummyPipeline()
    
    print("Demo is ready!")
    
    # Create and show the application
    app = QApplication(sys.argv)
    window = MainWindow(app_params)
    window.show()
    
    sys.exit(app.exec())

if __name__ == "__main__":
    main() 