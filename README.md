# Pix2Pix - Image-to-Image Translation with cGAN
This project implements Pix2Pix using Conditional GANs for tasks like edges→shoes, facades→maps etc.

## Steps to Run
1. Create virtual environment
2. Install dependencies: `pip install -r requirements.txt`
3. Train: `python train.py --dataroot ./data/facades --name facades_pix2pix --model pix2pix`
4. Test: `python test.py --dataroot ./data/facades --name facades_pix2pix --model pix2pix`
