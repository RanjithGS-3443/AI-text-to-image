# Text-to-Image Generation using Diffusers Library

# Install the required libraries before running this script
# pip install diffusers torch transformers

from diffusers import StableDiffusionPipeline
import torch

# Check if GPU is available and set the device accordingly
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the pre-trained Stable Diffusion pipeline
pipeline = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
pipeline = pipeline.to(device)

# Function to generate an image from text
def generate_image(prompt, output_path="generated_image.png"):
    """
    Generate an image based on the provided text prompt.

    Args:
        prompt (str): The text prompt for image generation.
        output_path (str): The file path to save the generated image.

    Returns:
        None
    """
    print(f"Generating image for prompt: '{prompt}'")
    
    # Generate the image
    image = pipeline(prompt).images[0]

    # Save the generated image
    image.save(output_path)
    print(f"Image saved at: {output_path}")

# Example usage
if __name__ == "__main__":
    # Define your text prompt here
    text_prompt = "horse standing in highway"

    # Specify the output file name
    output_file = "horse_image.png"

    # Call the function to generate the image
    generate_image(text_prompt, output_file)
