import torch
import gradio as gr
from diffusers import DDIMScheduler, DDIMInverseScheduler, StableDiffusionDiffEditPipeline
from diffusers.utils import load_image, make_image_grid
from PIL import Image
from transformers import AutoTokenizer, T5ForConditionalGeneration

# Initialize T5 model for prompt generation
def init_t5_model():
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
    model = T5ForConditionalGeneration.from_pretrained(
        "google/flan-t5-large",
        torch_dtype=torch.float32,
        device_map="cpu"
    )
    return tokenizer, model

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
    pipeline.to("cpu")
    return pipeline

@torch.no_grad()
def generate_prompts(input_prompt, tokenizer, model):
    input_ids = tokenizer(input_prompt, return_tensors="pt").input_ids.to("cpu")
    outputs = model.generate(
        input_ids,
        temperature=0.8,
        num_return_sequences=16,
        do_sample=True,
        max_new_tokens=128,
        top_k=10
    )
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)

@torch.no_grad()
def embed_prompts(sentences, tokenizer, text_encoder):
    embeddings = []
    for sent in sentences:
        text_inputs = tokenizer(
            sent,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        prompt_embeds = text_encoder(text_input_ids.to("cpu"), attention_mask=None)[0]
        embeddings.append(prompt_embeds)
    return torch.concatenate(embeddings, dim=0).mean(dim=0).unsqueeze(0)

# Process the image with regular prompts
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

# Process the image with embeddings
def process_image_with_embeddings(image, source_concept, target_concept):
    # Resize image to 768x768
    image = image.resize((768, 768))
    
    # Generate prompts using T5
    source_text = f"Provide a caption for images containing a {source_concept}. The captions should be in English and should be no longer than 150 characters."
    target_text = f"Provide a caption for images containing a {target_concept}. The captions should be in English and should be no longer than 150 characters."
    
    source_prompts = generate_prompts(source_text, t5_tokenizer, t5_model)
    target_prompts = generate_prompts(target_text, t5_tokenizer, t5_model)
    
    # Generate embeddings
    source_embeds = embed_prompts(source_prompts, pipeline.tokenizer, pipeline.text_encoder)
    target_embeds = embed_prompts(target_prompts, pipeline.tokenizer, pipeline.text_encoder)
    
    # Generate mask
    mask_image = pipeline.generate_mask(
        image=image,
        source_prompt_embeds=source_embeds,
        target_prompt_embeds=target_embeds,
    )
    
    # Invert latents
    inv_latents = pipeline.invert(
        prompt_embeds=source_embeds,
        image=image,
    ).latents
    
    # Generate output image
    output_image = pipeline(
        mask_image=mask_image,
        image_latents=inv_latents,
        prompt_embeds=target_embeds,
        negative_prompt_embeds=source_embeds,
    ).images[0]
    
    # Convert mask to PIL Image
    mask_image = Image.fromarray((mask_image.squeeze()*255).astype("uint8"), "L").resize((768, 768))
    
    # Create grid of images
    grid = make_image_grid([image, mask_image, output_image], rows=1, cols=3)
    return grid

# Initialize models
pipeline = init_pipeline()
t5_tokenizer, t5_model = init_t5_model()

# Create Gradio interface
with gr.Blocks(title="DiffEdit Image Editor") as demo:
    gr.Markdown("# DiffEdit Image Editor")
    gr.Markdown("Upload an image and specify source and target prompts to edit it.")
    
    with gr.Tabs():
        with gr.TabItem("Regular Prompts"):
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
        
        with gr.TabItem("Target Embeddings"):
            with gr.Row():
                with gr.Column():
                    input_image_emb = gr.Image(type="pil", label="Input Image")
                    source_concept = gr.Textbox(label="Source Concept", placeholder="e.g., bowl")
                    target_concept = gr.Textbox(label="Target Concept", placeholder="e.g., basket")
                    process_emb_btn = gr.Button("Process Image with Embeddings")
                
                with gr.Column():
                    output_image_emb = gr.Image(label="Result (Original | Mask | Edited)")
            
            process_emb_btn.click(
                fn=process_image_with_embeddings,
                inputs=[input_image_emb, source_concept, target_concept],
                outputs=output_image_emb
            )

if __name__ == "__main__":
    demo.launch() 