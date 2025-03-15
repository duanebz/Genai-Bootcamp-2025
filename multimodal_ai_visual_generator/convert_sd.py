from optimum.intel.openvino import OVStableDiffusionPipeline
from diffusers import LMSDiscreteScheduler
import shutil
from pathlib import Path

print("Starting Stable Diffusion model conversion with LMS scheduler...")

# Load the model from Hugging Face
model_id = "runwayml/stable-diffusion-v1-5"
print(f"Loading model {model_id} from Hugging Face...")

# Create the output directory if it doesn't exist
output_dir = Path("models/stable-diffusion-ov")
if output_dir.exists():
    print(f"Output directory {output_dir} already exists, removing it...")
    shutil.rmtree(output_dir)

# Create a pipeline with LMS scheduler
print("Creating pipeline with LMS scheduler...")
scheduler = LMSDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")

# Convert to OpenVINO format
print(f"Converting model to OpenVINO format and saving to {output_dir}...")
ov_pipe = OVStableDiffusionPipeline.from_pretrained(
    model_id, 
    scheduler=scheduler,
    export=True
)

# Save the converted model
ov_pipe.save_pretrained(str(output_dir))
print(f"Model successfully converted and saved to {output_dir}") 