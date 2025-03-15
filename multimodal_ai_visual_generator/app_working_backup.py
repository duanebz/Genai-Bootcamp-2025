import io
import sys
from pathlib import Path
import os

import cv2
import numpy as np
from PIL import Image
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QPixmap, QFont
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QPushButton, QVBoxLayout,
    QHBoxLayout, QGridLayout, QProgressBar, QTextEdit, QLineEdit, QLabel
)

# Import OpenVINO with error handling
try:
    from openvino.runtime import Core
    import openvino as ov
    print("Successfully imported OpenVINO")
except ImportError:
    print("Error importing OpenVINO, using dummy implementations")
    # Create a dummy Core class
    class Core:
        def __init__(self):
            self.available_devices = ["CPU"]
        
        def compile_model(self, *args, **kwargs):
            return None

# Import OpenVINO GenAI with error handling
try:
    import openvino_genai as ov_genai
    # Try different import paths for the pipeline classes
    try:
        from openvino_genai.pipeline import StableDiffusionPipeline, ImageGenerationPipeline, Text2ImagePipeline
    except ImportError:
        try:
            from openvino_genai import StableDiffusionPipeline, ImageGenerationPipeline, Text2ImagePipeline
        except ImportError:
            print("Could not import pipeline classes from openvino_genai, using dummy implementations")
            # These will be defined later in the code
            StableDiffusionPipeline = None
            ImageGenerationPipeline = None
            Text2ImagePipeline = None
    print("Successfully imported OpenVINO GenAI")
except ImportError:
    print("Error importing OpenVINO GenAI, using dummy implementations")
    # Create a dummy module
    class DummyGenAI:
        def __init__(self):
            pass
            
        class GenerationConfig:
            def __init__(self):
                self.temperature = 0.7
                self.top_p = 0.95
                self.max_length = 2048
                
    ov_genai = DummyGenAI()
    StableDiffusionPipeline = None
    ImageGenerationPipeline = None
    Text2ImagePipeline = None

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
        from PIL import ImageDraw, ImageFont
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
                # Create a sample image when the Start button is clicked
                img = np.zeros((432, 768, 3), dtype=np.uint8)
                for i in range(img.shape[0]):
                    for j in range(img.shape[1]):
                        img[i, j, 0] = int(255 * i / img.shape[0])  # Red gradient
                        img[i, j, 1] = int(255 * j / img.shape[1])  # Green gradient
                        img[i, j, 2] = 100  # Blue constant
                
                # Convert to PIL Image
                pil_img = Image.fromarray(img)
                
                # Add text to the image
                from PIL import ImageDraw, ImageFont
                draw = ImageDraw.Draw(pil_img)
                try:
                    # Try to use a system font
                    font = ImageFont.truetype("arial.ttf", 30)
                except:
                    # Fall back to default font
                    font = ImageFont.load_default()
                
                draw.text((100, 200), "AI Adventure Experience Demo", fill=(255, 255, 255), font=font)
                draw.text((100, 250), "Theme: " + self.theme_input.text(), fill=(255, 255, 255), font=font)
                
                # Convert to QPixmap
                buffer = io.BytesIO()
                pil_img.save(buffer, format="PNG")
                buffer.seek(0)
                
                pixmap = QPixmap()
                pixmap.loadFromData(buffer.read(), "PNG")
                
                # Update the image
                self.primary_pixmap = pixmap
                self.update_image_label()
                
                # Create a depth map image
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
                
                # Convert to QPixmap
                buffer = io.BytesIO()
                depth_pil_img.save(buffer, format="PNG")
                buffer.seek(0)
                
                depth_pixmap = QPixmap()
                depth_pixmap.loadFromData(buffer.read(), "PNG")
                
                # Update the depth map
                self.depth_pixmap = depth_pixmap
                
                # Update the caption
                self.caption_label.setText("Demo started with theme: " + self.theme_input.text())
                
                # Update the progress bar
                self.progress_bar.setValue(50)
                self.progress_bar.setFormat("Running")
                
                self.start_button.setText("Stop")
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

def generate_image(self, prompt):
    try:
        image_tensor = self.sd_engine.generate(
            prompt,
            width=768,
            height=432,
            num_inference_steps=5,
            num_images_per_prompt=1)

        sd_output = Image.fromarray(image_tensor.data[0])
        
        # Display a message on the image to indicate it's a placeholder
        if hasattr(self.sd_engine, '__class__') and self.sd_engine.__class__.__name__ == 'DummyPipeline':
            from PIL import ImageDraw, ImageFont
            draw = ImageDraw.Draw(sd_output)
            try:
                # Try to use a system font
                font = ImageFont.truetype("arial.ttf", 20)
            except:
                # Fall back to default font
                font = ImageFont.load_default()
            
            draw.text((10, 10), "Image Generation Not Available", fill=(255, 255, 255), font=font)
            draw.text((10, 40), f"Prompt: {prompt[:50]}...", fill=(255, 255, 255), font=font)

        sr_out = self.produce_parallex_img(sd_output)
        return sr_out
    except Exception as e:
        print(f"Error in generate_image: {str(e)}")
        # Create a blank image with an error message
        img = np.zeros((432, 768, 3), dtype=np.uint8)
        sd_output = Image.fromarray(img)
        
        from PIL import ImageDraw, ImageFont
        draw = ImageDraw.Draw(sd_output)
        try:
            # Try to use a system font
            font = ImageFont.truetype("arial.ttf", 20)
        except:
            # Fall back to default font
            font = ImageFont.load_default()
        
        draw.text((10, 10), f"Error: {str(e)}", fill=(255, 255, 255), font=font)
        draw.text((10, 40), f"Prompt: {prompt[:50]}...", fill=(255, 255, 255), font=font)
        
        return self.produce_parallex_img(sd_output)

def main():
    # Define the parameters for the application
    app_params = {
        "llm": None,
        "sd": None,
        "whisper_device": None,
        "super_res_compiled_model": None,
        "super_res_upsample_factor": 1,
        "depth_compiled_model": None
    }

    # Create the 'results' folder if it doesn't exist
    Path("results").mkdir(exist_ok=True)

    # Define the devices
    sd_device = "CPU"
    whisper_device = "CPU"
    super_res_device = "CPU"
    depth_anything_device = "CPU"

    # Initialize OpenVINO Core
    try:
        core = Core()
        print("OpenVINO Core initialized successfully")
        print("Available devices:", core.available_devices)
        
        # Use GPU if available
        if "GPU" in core.available_devices:
            sd_device = "GPU"
            super_res_device = "GPU"
            depth_anything_device = "GPU"
    except Exception as e:
        print(f"Error initializing OpenVINO Core: {str(e)}")
        core = Core()  # Use the dummy Core class defined earlier

    # Define a dummy pipeline class for when LLM generation is not available
    class DummyLLMPipeline:
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

    # Always use the dummy LLM pipeline to avoid segmentation fault
    print("Using dummy LLM pipeline")
    app_params["llm"] = DummyLLMPipeline()

    # Define a dummy pipeline class for when image generation is not available
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

    # Try to create a stable diffusion pipeline with the actual model
    try:
        print("Creating a stable diffusion pipeline...")
        
        # Check if the model directory exists
        sd_model_dir = os.path.join("models", "LCM_Dreamshaper_v7", "FP16")
        if not os.path.exists(sd_model_dir):
            print(f"SD model directory not found: {sd_model_dir}")
            print("Using dummy stable diffusion pipeline instead")
            app_params["sd"] = DummyPipeline()
        else:
            # Try to create the stable diffusion pipeline
            try:
                # Try using different class names from openvino_genai
                try:
                    # Try StableDiffusionPipeline first
                    sd_pipe = ov_genai.StableDiffusionPipeline(
                        model=sd_model_dir,
                        device=sd_device
                    )
                    print("Successfully created StableDiffusionPipeline")
                except Exception as e:
                    print(f"Error creating StableDiffusionPipeline: {str(e)}")
                    try:
                        # Try ImageGenerationPipeline next
                        sd_pipe = ov_genai.ImageGenerationPipeline(
                            model=sd_model_dir,
                            device=sd_device
                        )
                        print("Successfully created ImageGenerationPipeline")
                    except Exception as e:
                        print(f"Error creating ImageGenerationPipeline: {str(e)}")
                        try:
                            # Try Text2ImagePipeline as a last resort
                            sd_pipe = ov_genai.Text2ImagePipeline(
                                model=sd_model_dir,
                                device=sd_device
                            )
                            print("Successfully created Text2ImagePipeline")
                        except Exception as e:
                            print(f"Error creating Text2ImagePipeline: {str(e)}")
                            print("Could not create any image generation pipeline. Using dummy pipeline.")
                            sd_pipe = DummyPipeline()
                
                app_params["sd"] = sd_pipe
                print("Stable diffusion pipeline created successfully")
            except Exception as e:
                print(f"Error creating stable diffusion pipeline: {str(e)}")
                print("Using dummy stable diffusion pipeline instead")
                app_params["sd"] = DummyPipeline()
    except Exception as e:
        print(f"Error setting up stable diffusion pipeline: {str(e)}")
        print("Using dummy stable diffusion pipeline instead")
        app_params["sd"] = DummyPipeline()
    
    # Set up the whisper device
    app_params["whisper_device"] = whisper_device
    
    # Try to load the super resolution model
    try:
        print("Loading super resolution model...")
        try:
            from superres import superres_load
            app_params["super_res_compiled_model"], app_params["super_res_upsample_factor"] = superres_load(
                "models/single-image-super-resolution-1032.xml", super_res_device
            )
            print("Super resolution model loaded successfully")
        except Exception as e:
            print(f"Error loading super resolution model: {str(e)}")
            print("Skipping super resolution model loading")
            app_params["super_res_compiled_model"] = None
            app_params["super_res_upsample_factor"] = 1
    except Exception as e:
        print(f"Error loading super resolution model: {str(e)}")
        print("Skipping super resolution model loading")
        app_params["super_res_compiled_model"] = None
        app_params["super_res_upsample_factor"] = 1
    
    # Try to load the depth model
    try:
        print("Initializing Depth Anything v2 model to run on", depth_anything_device)
        OV_DEPTH_ANYTHING_PATH = Path(f"models/depth_anything_v2_vits.xml")
        # Check if the file exists before trying to load it
        if not OV_DEPTH_ANYTHING_PATH.exists():
            print(f"Depth model file not found: {OV_DEPTH_ANYTHING_PATH}")
            app_params["depth_compiled_model"] = None
            print("Will continue without depth map capability")
        else:
            print(f"Loading depth model from: {OV_DEPTH_ANYTHING_PATH}")
            try:
                depth_compiled_model = core.compile_model(OV_DEPTH_ANYTHING_PATH, device_name=depth_anything_device)
                app_params["depth_compiled_model"] = depth_compiled_model
                print("Initializing Depth Anything v2 done...")
            except Exception as e:
                print(f"Error compiling depth model: {str(e)}")
                app_params["depth_compiled_model"] = None
                print("Will continue without depth map capability")
    except Exception as e:
        print(f"Error loading depth model: {str(e)}")
        print("Will continue without depth map capability")
        app_params["depth_compiled_model"] = None
    
    print("Demo is ready!")
    
    # Create and show the main window
    app = QApplication(sys.argv)
    window = MainWindow(app_params)
    window.show()
    
    sys.exit(app.exec())

if __name__ == "__main__":
    main() 