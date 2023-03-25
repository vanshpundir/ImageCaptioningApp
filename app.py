from flask import Flask, render_template, request
from models.blip import blip_decoder
import torch
from PIL import Image
import requests
import torch
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_demo_image(image_size, device, image_file):
    raw_image = Image.open(image_file).convert('RGB')
    w, h = raw_image.size
    raw_image.resize((w//5,h//5)).show()
    
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    ]) 
    image = transform(raw_image).unsqueeze(0).to(device)
    return image



    


app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the uploaded image file
    image_file = request.files['image']
    
    image = load_demo_image(384, device, image_file)
    model_url = 'model_base.pth'
    model = blip_decoder(pretrained=model_url, image_size=384, vit='base')
    model.eval()
    model = model.to(device)

    with torch.no_grad():
        # beam search
        caption = model.generate(image, sample=False, num_beams=3, max_length=20, min_length=5) 
        # nucleus sampling
        # caption = model.generate(image, sample=True, top_p=0.9, max_length=20, min_length=5) 
        result = caption[0]
    # Render the results page with the output image
    return render_template('result.html', result=result)
