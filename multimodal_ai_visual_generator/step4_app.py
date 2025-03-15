import io
import sys
from pathlib import Path
import os

import numpy as np
from PIL import Image
import cv2
# Import OpenVINO and OpenVINO GenAI
import openvino as ov
import openvino_genai as ov_genai
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPixmap, QFont
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QPushButton, QVBoxLayout,
    QHBoxLayout, QGridLayout, QProgressBar, QTextEdit, QLineEdit, QLabel
)

class ClickableLabel(QLabel):
    clicked = Signal()  # Define a signal to emit on click

    def mousePressEvent(self, event):
        self.clicked.emit()  # Emit the clicked signal
        super().mousePressEvent(event)

class MainWindow(QMainWindow):
    def __init__(self, app_params):
        super().__init__()

        # Store app parameters
        self.app_params = app_params
        self.llm_pipeline = app_params.get("llm")
        self.sd_engine = app_params.get("sd")

        # Main widget and layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        layout = QGridLayout(self.central_widget)

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
        self.start_button.clicked.connect(self.start_demo)
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

    def start_demo(self):
        try:
            # Try to use the LLM pipeline if available
            if self.llm_pipeline:
                try:
                    # Create a simple prompt
                    prompt = "Generate a short description of a fantasy scene."
                    
                    # Update the progress bar
                    self.progress_bar.setValue(25)
                    self.progress_bar.setFormat("Generating text...")
                    
                    # Generate text using the LLM pipeline
                    result = self.llm_pipeline.generate(prompt)
                    
                    # Update the caption
                    self.caption_label.setText(result)
                    
                    # Try to generate an image if the SD engine is available
                    if self.sd_engine:
                        try:
                            # Update the progress bar
                            self.progress_bar.setValue(50)
                            self.progress_bar.setFormat("Generating image...")
                            
                            # Generate image using the SD engine
                            image_tensor = self.sd_engine.generate(
                                result,
                                width=768,
                                height=432,
                                num_inference_steps=5,
                                num_images_per_prompt=1
                            )
                            
                            # Convert the image tensor to a PIL Image
                            sd_output = Image.fromarray(image_tensor.data[0])
                            
                            # Convert to QPixmap
                            buffer = io.BytesIO()
                            sd_output.save(buffer, format="PNG")
                            buffer.seek(0)
                            
                            pixmap = QPixmap()
                            pixmap.loadFromData(buffer.read(), "PNG")
                            
                            # Update the image
                            self.primary_pixmap = pixmap
                            self.update_image_label()
                            
                            # Update the progress bar
                            self.progress_bar.setValue(100)
                            self.progress_bar.setFormat("Image generated")
                        except Exception as e:
                            print(f"Error using SD engine: {str(e)}")
                            self.log_widget.append(f"Error using SD engine: {str(e)}")
                            
                            # Fall back to the default behavior
                            self.create_default_image()
                    else:
                        # Fall back to the default behavior
                        self.create_default_image()
                except Exception as e:
                    print(f"Error using LLM pipeline: {str(e)}")
                    self.caption_label.setText(f"Error using LLM: {str(e)}")
                    
                    # Fall back to the default behavior
                    self.create_default_image()
            else:
                # Fall back to the default behavior
                self.create_default_image()
            
            if self.start_button.text() == "Start":
                self.start_button.setText("Stop")
            else:
                self.start_button.setText("Start")
                self.progress_bar.setValue(0)
                self.progress_bar.setFormat("Idle")
        except Exception as e:
            print(f"Error starting demo: {str(e)}")
            self.caption_label.setText(f"Error: {str(e)}")
    
    def create_default_image(self):
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
        
        # Update the caption
        self.caption_label.setText("Demo started with theme: " + self.theme_input.text())
        
        # Update the progress bar
        self.progress_bar.setValue(50)
        self.progress_bar.setFormat("Running")

    def toggle_theme(self):
        if self.theme_input.isVisible():
            self.theme_input.hide()
        else:
            self.theme_input.show()

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
        if self.display_primary_img:
            self.caption_label.setText("Showing main image")
        else:
            self.caption_label.setText("Showing depth map")

def main():
    app = QApplication(sys.argv)
    
    print("Starting step 4 demo...")
    
    # Create the 'results' folder if it doesn't exist
    Path("results").mkdir(exist_ok=True)
    
    # Initialize OpenVINO Core
    try:
        core = ov.Core()
        print("OpenVINO Core initialized successfully")
        print("Available devices:", core.available_devices)
    except Exception as e:
        print(f"Error initializing OpenVINO Core: {str(e)}")
        core = None
    
    app_params = {}
    
    # Try to create an LLM pipeline with the actual model
    try:
        print("Creating an LLM pipeline...")
        
        # Check if the model directory exists
        model_dir = os.path.join("models", "llama-3.2-3b-instruct-openvino")
        if not os.path.exists(model_dir):
            print(f"LLM model directory not found: {model_dir}")
            raise FileNotFoundError(f"LLM model directory not found: {model_dir}")
        
        # Try to create the LLM pipeline
        try:
            config = {
                "PERFORMANCE_HINT": "LATENCY"
            }
            
            llm_pipe = ov_genai.LLMPipeline(
                model_path=model_dir,
                device="CPU",
                config=config
            )
            app_params["llm"] = llm_pipe
            print("LLM pipeline created successfully")
        except Exception as e:
            print(f"Error creating LLM pipeline: {str(e)}")
            raise
    except Exception as e:
        print(f"Error setting up LLM pipeline: {str(e)}")
        print("Using dummy LLM pipeline instead")
        
        # Define a dummy LLM pipeline class
        class DummyLLMPipeline:
            def generate(self, prompt):
                print(f"Generating text for prompt: {prompt}")
                return "SD Prompt: A beautiful landscape with mountains and a lake."
            
            def get_tokenizer(self):
                class DummyTokenizer:
                    def apply_chat_template(self, *args, **kwargs):
                        return ""
                return DummyTokenizer()
        
        # Use the dummy LLM pipeline
        app_params["llm"] = DummyLLMPipeline()
        print("Using dummy LLM pipeline")
    
    # Try to create a stable diffusion pipeline with the actual model
    try:
        print("Creating a stable diffusion pipeline...")
        
        # Check if the model directory exists
        sd_model_dir = os.path.join("models", "LCM_Dreamshaper_v7", "FP16")
        if not os.path.exists(sd_model_dir):
            print(f"SD model directory not found: {sd_model_dir}")
            raise FileNotFoundError(f"SD model directory not found: {sd_model_dir}")
        
        # Try to create the stable diffusion pipeline
        try:
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
            
            # Try using different class names from openvino_genai
            try:
                # Try StableDiffusionPipeline first
                sd_pipe = ov_genai.StableDiffusionPipeline(
                    model=sd_model_dir,
                    device="CPU"
                )
                print("Successfully created StableDiffusionPipeline")
            except Exception as e:
                print(f"Error creating StableDiffusionPipeline: {str(e)}")
                try:
                    # Try ImageGenerationPipeline next
                    sd_pipe = ov_genai.ImageGenerationPipeline(
                        model=sd_model_dir,
                        device="CPU"
                    )
                    print("Successfully created ImageGenerationPipeline")
                except Exception as e:
                    print(f"Error creating ImageGenerationPipeline: {str(e)}")
                    try:
                        # Try Text2ImagePipeline as a last resort
                        sd_pipe = ov_genai.Text2ImagePipeline(
                            model=sd_model_dir,
                            device="CPU"
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
            raise
    except Exception as e:
        print(f"Error setting up stable diffusion pipeline: {str(e)}")
        print("Using dummy stable diffusion pipeline instead")
        
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
        
        # Use the dummy stable diffusion pipeline
        app_params["sd"] = DummyPipeline()
        print("Using dummy stable diffusion pipeline")
    
    window = MainWindow(app_params)
    window.show()
    
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
