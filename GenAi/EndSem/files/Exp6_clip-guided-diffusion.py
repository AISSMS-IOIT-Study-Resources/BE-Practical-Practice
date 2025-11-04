import torch
from diffusers import StableDiffusionPipeline
import matplotlib.pyplot as plt
from cleanfid import fid
import os

def generate_simple_image(prompt, output_dir="images", steps=30):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Simple filename based on prompt
    safe_prompt = "".join(c for c in prompt if c.isalnum() or c in (' ', '-', '_')).rstrip()
    safe_prompt = safe_prompt.replace(' ', '_')[:30]  # Limit length
    filename = f"{safe_prompt}.png"
    output_path = os.path.join(output_dir, filename)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = StableDiffusionPipeline.from_pretrained(
        "stable-diffusion-v1-5/stable-diffusion-v1-5", 
        torch_dtype=torch.float16 if device=="cuda" else torch.float32,
        safety_checker=None
    )
    pipe = pipe.to(device)
    image = pipe(prompt, num_inference_steps=steps).images[0]
    image.save(output_path)
    
    print(f"Image saved to: {output_path}")
    
    # Display the generated image
    plt.figure(figsize=(8, 8))
    plt.imshow(image)
    plt.axis('off')
    plt.title(f"Generated: {prompt}")
    plt.show()
    
    return image, output_path, output_dir

def evaluate_with_fid(image_dir, reference='cifar10'):
    # Compute FID against CIFAR-10 as reference
    fid_score = fid.compute_fid(image_dir, reference)
    print(f"FID Score (vs {reference}): {fid_score}")
    
    # Also compute Inception Score for quality
    is_score = fid.compute_is(image_dir)
    print(f"Inception Score: {is_score}")
    
    return fid_score, is_score

prompt = "an astronaut bear on mars"
image, path, directory = generate_simple_image(prompt)

# Evaluate the generated image using FID
print("\nEvaluating generated image...")
fid_score, is_score = evaluate_with_fid(directory)