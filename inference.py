from diffusers import StableDiffusionPipeline
import torch

def main():
    model_id = "textual_inversion_cat"
    pipe = StableDiffusionPipeline.from_pretrained(model_id,torch_dtype=torch.float16).to("cuda")
    prompt = "A <cat-toy> backpack"

    image = pipe(prompt, num_inference_steps=50, guidance_scale=7.5).images[0]

    image.save("cat-backpack.png")

if __name__ == "__main__":
    main()