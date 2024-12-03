import gradio as gr
import json
from transformers import pipeline
from PIL import Image
import requests

# Load bad words
with open("bad_words.json", "r") as file:
    bad_words = set(json.load(file))

# Load Hugging Face pipeline
model_id = "CompVis/stable-diffusion-v1-4"
text_to_image = pipeline("text-to-image-generation", model=model_id)

def generate_image(prompt):
    # Check for bad words
    for word in bad_words:
        if word in prompt.lower():
            return "Prompt contains inappropriate content. Please try again."
    
    # Generate the image
    try:
        result = text_to_image(prompt)
        image_url = result[0]['image']
        image = Image.open(requests.get(image_url, stream=True).raw)
        return image
    except Exception as e:
        return f"Error generating image: {e}"

# Gradio Interface
interface = gr.Interface(
    fn=generate_image,
    inputs=gr.Textbox(label="Enter your prompt"),
    outputs="image",
    title="Text-to-Image Generator",
    description="Generate an image from a text description using Hugging Face Stable Diffusion.",
    examples=["A futuristic city skyline", "A serene mountain landscape"],
)

if __name__ == "__main__":
    interface.launch(share=True)
