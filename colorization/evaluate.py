
import argparse
import matplotlib.pyplot as plt

from colorizers import *

parser = argparse.ArgumentParser()
parser.add_argument('-i','--img_path', type=str, default='imgs/ansel_adams3.jpg')
parser.add_argument('--use_gpu', action='store_true', help='whether to use GPU')
parser.add_argument('-o','--save_prefix', type=str, default='saved', help='will save into this file with {places365.png, imagenet.png} suffixes')
opt = parser.parse_args()

# load colorizers
colorizer_places365 = ECCVGenerator()
checkpoint = torch.load('./colorization_model.pth')
colorizer_places365.load_state_dict(checkpoint)
colorizer_places365.eval()

colorizer_imagenet = eccv16(pretrained=True).eval()
if(opt.use_gpu):
	colorizer_places365.cuda()
	colorizer_imagenet.cuda()

# default size to process images is 256x256
# grab L channel in both original ("orig") and resized ("rs") resolutions
img = load_img(opt.img_path)
(tens_l_orig, tens_l_rs) = preprocess_img(img, HW=(256,256))
if(opt.use_gpu):
	tens_l_rs = tens_l_rs.cuda()

# colorizer outputs 256x256 ab map
# resize and concatenate to original L channel
img_bw = postprocess_tens(tens_l_orig, torch.cat((0*tens_l_orig,0*tens_l_orig),dim=1))
out_img_places365 = postprocess_tens(tens_l_orig, colorizer_places365(tens_l_rs).cpu())
out_img_imagenet = postprocess_tens(tens_l_orig, colorizer_imagenet(tens_l_rs).cpu())

plt.imsave('%s_places365.png'%opt.save_prefix, out_img_places365)
plt.imsave('%s_imagenet.png'%opt.save_prefix, out_img_imagenet)

plt.figure(figsize=(12,8))
plt.subplot(2,2,1)
plt.imshow(img)
plt.title('Original')
plt.axis('off')

plt.subplot(2,2,2)
plt.imshow(img_bw)
plt.title('Input')
plt.axis('off')

plt.subplot(2,2,3)
plt.imshow(out_img_places365)
plt.title('Output (Places365)')
plt.axis('off')

plt.subplot(2,2,4)
plt.imshow(out_img_imagenet)
plt.title('Output (Imagenet)')
plt.axis('off')
plt.show()
