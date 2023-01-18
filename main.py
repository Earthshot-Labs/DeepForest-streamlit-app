# python3 -m venv venv
# . venv/bin/activate
# pip install streamlit
# pip install torch torchvision
# streamlit run main.py
import streamlit as st
from PIL import Image
import tifffile
from networks import Generator
from option import InferenceOption
import torch
from pipeline import CustomDataset
from utils import Manager
import glob
import numpy as np
from skimage.morphology import erosion, dilation, opening, closing
import os

# inspired by: https://github.com/python-engineer/pytorch-examples/tree/master/streamlit-style-transfer/neural_style
def multi_dil(im, num):
    for i in range(num):
        im = dilation(im)
    return im

def multi_ero(im, num):
    for i in range(num):
        im = erosion(im)
    return im

def post_uploads(stack, image_name):
    print(f'Saving image in content-images: {image_name} -- image shape: {stack.shape}')
    if not os.path.exists("images/content-images/"):
        os.makedirs("images/content-images/")
    if not os.path.exists("images/output-images"):
        os.makedirs("images/output-images")
    tifffile.imwrite(f"images/content-images/{image_name}", stack)

    # To View Uploaded Image
    image = tifffile.imread(f"images/content-images/{image_name}")
    # save img in right folder
    tifffile.imwrite(f"images/content-images/{image_name}", image)

    opt = InferenceOption().parse()

    st.write('### Source image with reforestation mask:')
    mask = stack[:, :, -1]
    image_rgb = stack[:, :, 0:3]
    st.image([image_rgb, mask], width=300) #image: numpy array
    clicked = st.button('Reforest!')
    print(f'stack: {stack.shape}')

    if clicked:
        # if not os.path.exists(f"images/output-images/{image_name}"):
        G = Generator(opt)

        dataset = CustomDataset(opt)
        data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=1,
                                                  shuffle=False, num_workers=2)
        G.load_state_dict(torch.load(model, map_location=torch.device('cpu')))
        manager = Manager(opt)

        for i, input in enumerate(data_loader):
            print(images_input[i])
            input = input[0]
            fake = G(input)
            manager.save_image(fake.detach(), f"images/output-images/{images_input[i]}")
            manager.save_image_overlay(fake.detach(), input.detach(),  f"images/output-images/{images_input[i].replace('.tif', 'v2.tif')}")

        st.write('### Reforested image (cropping non reforested areas):')
        image = Image.open(f"images/output-images/{image_name.replace('.tif', 'v2.tif')}")
        tifffile.imwrite(f"images/output-images/{image_name.replace('.tif', '.png')}", np.array(image)[:, :, 0:3])
        st.image(image, width=800)

        # st.write('### Reforested image:')
        # image = Image.open(f"images/output-images/{image_name}")
        # img_fixed_color = manager.match_colors(np.array(image), image_rgb)
        # st.image(img_fixed_color, width=300)



st.title('Climate Visualization: Reforestation')
st.write('By Earthshot Labs')

# TODO
# Cleaning folder for faster inference in web app
# if os.path.exists("images/content-images"):
#     os.remove("images/content-images")
#     os.makedirs("images/content-images")
# else:
#     os.makedirs("images/content-images")

images_input = glob.glob("images/content-images/*.tif")
images_input.sort()
images_input = [path.split('/')[-1] for path in images_input]

# models = glob.glob("models/**/*.pt")
# model = st.sidebar.selectbox(
#     'Select Model',
#     (models)
# )
model = st.sidebar.file_uploader("Upload Model here", type=["pt"])

if model is not None:
    image_file = st.sidebar.file_uploader("Upload Image", type=["jpg", "png", "tif"])
    if image_file is not None:
        image_name = image_file.name
        print(image_name)
        if '.tif' in image_name:
            print('tifffile -- not need for mask image')
            stack = tifffile.imread(image_file)
            if stack.shape[0:2] != (1024, 1024):
                st.write('Wrong image size -- cropping image to 1024x1024')
                print("Wrong image size -- cropping image")
                # TODO  center crop?
                # stack = cv2.resize(stack, (1024, 1024), cv2.INTER_NEAREST)
                stack = stack[0:1024, 0:1024, :]
            post_uploads(stack, image_name)
        else:
            mask_file = st.sidebar.file_uploader("Upload Image with reforestation label", type=["jpg", "png"])
            image = np.array(Image.open(image_file))
            if mask_file is not None:
                mask = np.array(Image.open(mask_file))
                print(f"Mask: {mask.shape}")
                if mask.shape[0:2] != (1024, 1024):
                    st.write('Wrong image size -- cropping image to 1024x1024')
                    if len(mask.shape) != 3:
                        mask = np.expand_dims(mask, axis=-1)
                    mask = mask[0:1024, 0:1024, :]
                    image = image[0:1024, 0:1024, :]
                mask_pixels = np.all(mask >= [255, 255, 255], axis=-1)
                non_mask_pixels = np.any(mask < [255, 255, 255], axis=-1)

                mask[mask_pixels] = [255, 255, 255]
                mask[non_mask_pixels] = [0, 0, 0]
                mask = multi_dil(mask, 5)
                mask = multi_ero(mask, 5)
                mask = mask[:, :, 0]
                assert np.unique(mask)[0] == 0

                stack = np.dstack((image, mask))
                image_name = image_name.replace('jpg', 'tif')
                print("Mask generated")
                post_uploads(stack, image_name)



