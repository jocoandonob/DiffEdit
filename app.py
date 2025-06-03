import torch
import gradio as gr
from diffusers import DDIMScheduler, DDIMInverseScheduler, StableDiffusionDiffEditPipeline
from diffusers.utils import load_image, make_image_grid
from PIL import Image

# Initialize the pipeline
def init_pipeline():
    pipeline = StableDiffusionDiffEditPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-1",
        torch_dtype=torch.float32,
        safety_checker=None,
        use_safetensors=True,
    )
    pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
    pipeline.inverse_scheduler = DDIMInverseScheduler.from_config(pipeline.scheduler.config)
    # Force CPU usage
    pipeline.to("cpu")
    return pipeline

# Process the image
def process_image(image, source_prompt, target_prompt):
    # Resize image to 768x768
    image = image.resize((768, 768))
    
    # Generate mask
    mask_image = pipeline.generate_mask(
        image=image,
        source_prompt=source_prompt,
        target_prompt=target_prompt,
    )
    
    # Invert latents
    inv_latents = pipeline.invert(prompt=source_prompt, image=image).latents
    
    # Generate output image
    output_image = pipeline(
        prompt=target_prompt,
        mask_image=mask_image,
        image_latents=inv_latents,
        negative_prompt=source_prompt,
    ).images[0]
    
    # Convert mask to PIL Image
    mask_image = Image.fromarray((mask_image.squeeze()*255).astype("uint8"), "L").resize((768, 768))
    
    # Create grid of images
    grid = make_image_grid([image, mask_image, output_image], rows=1, cols=3)
    return grid

# Initialize pipeline
pipeline = init_pipeline()

# Create Gradio interface
with gr.Blocks(title="DiffEdit Image Editor") as demo:
    gr.Markdown("# DiffEdit Image Editor")
    gr.Markdown("Upload an image and specify source and target prompts to edit it.")
    
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type="pil", label="Input Image")
            source_prompt = gr.Textbox(label="Source Prompt", placeholder="e.g., a bowl of fruits")
            target_prompt = gr.Textbox(label="Target Prompt", placeholder="e.g., a basket of pears")
            process_btn = gr.Button("Process Image")
        
        with gr.Column():
            output_image = gr.Image(label="Result (Original | Mask | Edited)")
    
    process_btn.click(
        fn=process_image,
        inputs=[input_image, source_prompt, target_prompt],
        outputs=output_image
    )

if __name__ == "__main__":
    demo.launch() 