import streamlit as st
import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

st.title("Transforming and Visualizing images in RGB and HSI Color Spaces")
def histogram_equalization(img, mode="rgb"):
    if mode=="rgb":
        ch = []
        for it in range(3):
            val, count = np.unique(img[:,:,it], return_counts=True)
            pdf = np.divide(count, sum(count)) 
            cdf = (val.max()*np.cumsum(pdf)).astype(np.int32)  
            mp = np.arange(0, val.max()+1)
            mp[val] = cdf  
            ch.append(mp[img[:,:,it]]) 
        heimage = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.int32) 
        heimage[:,:,0] = ch[0] 
        heimage[:,:,1] = ch[1] 
        heimage[:,:,2] = ch[2]  
        return heimage
    elif mode=="hsi":
        val, count = np.unique(img[:,:,-1], return_counts=True) 
        val = (val*255).astype(np.uint8)  
        pdf = np.divide(count, sum(count))  
        cdf = (val.max()*np.cumsum(pdf)).astype(np.int32) 
        mp = np.arange(0, val.max()+1)
        mp[val] = cdf  
        mul = np.multiply(img[:,:,-1], 255).astype(np.uint8) 
        img[:,:,-1] = mp[mul]  
        img[:,:,-1] = img[:,:,-1]/255  
        r, g, b = hsi2rgb(img)  
        heimage = np.zeros((img.shape[0], img.shape[1], 3)) 
        heimage[:,:,0] = r  
        heimage[:,:,1] = g
        heimage[:,:,2] = b  
        return heimage
    else:
        raise Exception("Please use either RGB/HSI mode")
def hsi2rgb(hsiimg):
    rows, cols = hsiimg[:,:,0].shape  
    h = hsiimg[:,:,0]  
    h = ((h - h.min()) * (1/(h.max() - h.min()) * 360)) 
    s = hsiimg[:,:,1]  
    i = hsiimg[:,:,2]  
    rd, gr, bl = [], [], []  
    for r in range(rows):
        for c in range(cols):
            if (h[r,c] >= 0 and h[r,c] <= 120):
                red = (1+((s[r, c]*np.cos(np.radians(h[r, c])))/np.cos(np.radians(60-h[r, c]))))/3
                blue = (1-s[r, c])/3
                rd.append(red)
                gr.append(1-(red+blue))
                bl.append(blue)
            elif (h[r,c] > 120 and h[r,c] <= 240):
                h[r, c] = h[r, c]-120
                red = (1-s[r, c])/3
                green = (1+((s[r, c]*np.cos(np.radians(h[r, c])))/np.cos(np.radians(60-h[r, c]))))/3
                rd.append(red)
                gr.append(green)
                bl.append(1-(red+green))
            elif (h[r,c] > 240 and h[r,c] <= 360):
                h[r, c] = h[r, c]-240
                green = (1-s[r, c])/3
                blue = (1+((s[r, c]*np.cos(np.radians(h[r, c])))/np.cos(np.radians(60-h[r, c]))))/3
                rd.append(1-(green+blue))
                gr.append(green)
                bl.append(blue)
    rd = np.multiply(rd, 3*i.flatten()).reshape(rows, cols)  
    gr = np.multiply(gr, 3*i.flatten()).reshape(rows, cols)
    bl = np.multiply(bl, 3*i.flatten()).reshape(rows, cols)
    return rd, gr, bl

def rgb2hsi(rgbimg):
    rows, cols = rgbimg.shape[:2]
    r = rgbimg[:,:,0] / 255.0
    g = rgbimg[:,:,1] / 255.0
    b = rgbimg[:,:,2] / 255.0
    h = np.zeros_like(r)
    s = np.zeros_like(r)
    i = np.zeros_like(r)
    for row in range(rows):
        for col in range(cols):
            num = 0.5 * ((r[row, col] - g[row, col]) + (r[row, col] - b[row, col]))
            den = np.sqrt((r[row, col] - g[row, col])**2 + (r[row, col] - b[row, col]) * (g[row, col] - b[row, col]))
            theta = np.arccos(np.clip(num / (den + 1e-8), -1, 1))
            if b[row, col] <= g[row, col]:
                h[row, col] = theta
            else:
                h[row, col] = 2 * np.pi - theta
            s[row, col] = 1 - 3 * np.minimum(np.minimum(r[row, col], g[row, col]), b[row, col]) / (r[row, col] + g[row, col] + b[row, col] + 1e-8)
            i[row, col] = (r[row, col] + g[row, col] + b[row, col]) / 3
    h = (h / (2 * np.pi)) * 360.0  
    return h, s, i
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
rgb_img = None
if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)
        rgb_img = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
        rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
        st.image(rgb_img, caption='Uploaded Image (RGB)',  width=300, use_column_width=False)
    except Exception as e:
        st.error("Error: Invalid image file.")
if rgb_img is not None:
    h, s, i = rgb2hsi(rgb_img)
    st.subheader("RGB to HSI Image Conversion H, S, I Values")
    col1, col2 = st.columns([2,1])
    
    with col1:
        st.write("Click this button to get H Value for the given image")
    with col2:
        if st.button("H Value"):
            st.write( np.mean(h))
    col1, col2 = st.columns([2,1])
    with col1:
        st.write("Click this button to get S Value for the given image")
    with col2:
        if st.button("S Value"):
            st.write( np.mean(s))
    col1, col2 = st.columns([2,1])
    with col1:
        st.write("Click this button to get I Value for the given image")
    with col2:
        if st.button("I Value"):
            st.write(np.mean(i))
    HSIimg = np.zeros((rgb_img.shape[0], rgb_img.shape[1], 3))
    HSIimg[:,:,0] = h
    HSIimg[:,:,1] = s
    HSIimg[:,:,2] = i
    r, g, b = hsi2rgb(HSIimg)
    RGBimg = np.zeros((rgb_img.shape[0], rgb_img.shape[1], 3))
    RGBimg[:,:,0] = r
    RGBimg[:,:,1] = g
    RGBimg[:,:,2] = b
    st.subheader("HSI to RGB Image Conversion R, G, B Values")
    col1, col2 = st.columns([2,1])
    with col1:
        st.write("Click this button to get R Value for the given image")
    with col2:
        if st.button("R Value"):
            st.write( np.mean(r))
    col1, col2 = st.columns([2,1])
    with col1:
        st.write("Click this button to get G Value for the given image")
    with col2:
        if st.button("G Value"):
            st.write( np.mean(g))
    col1, col2 = st.columns([2,1])
    with col1:
        st.write("Click this button to get B Value for the given image")
    with col2:
        if st.button("B Value"):
            st.write( np.mean(b))
    st.subheader("RGB to HSI Image Conversion")
    st.text("Click this button to convert input image to HSI components")
    if st.button("Hue, Saturation, Intensity component"):
        plt.figure(figsize=(15,5))
        plt.subplot(1,3,1)
        plt.imshow(h, cmap='gray')
        plt.title("Hue Component")
        plt.subplot(1,3,2)
        plt.imshow(s, cmap='gray')
        plt.title("Saturation Component")
        plt.subplot(1,3,3)
        plt.imshow(i, cmap='gray')
        plt.title("Intensity Component")
        st.pyplot(plt)
    st.subheader("HSI to RGB Image Conversion")
    st.text("Click this button to convert HSI image to RGB components")
    if st.button("Red, Green, Blue Component"):
        plt.figure(figsize=(15,5))
        plt.subplot(1,3,1)
        plt.imshow(r, cmap='gray')
        plt.title("Red Component")
        plt.subplot(1,3,2)
        plt.imshow(g, cmap='gray')
        plt.title("Green Component")
        plt.subplot(1,3,3)
        plt.imshow(b, cmap='gray')
        plt.title("Blue Component")
        st.pyplot(plt)
    st.subheader("Original RGB Image")
    st.text("Click this button to convert get original image")
    if st.button("Original RGB image"):
        plt.figure(figsize=(15,5))
        plt.subplot(1,3,1)
        plt.imshow(rgb_img)
        plt.title("Original RGB Image")
        st.pyplot(plt)

    st.subheader("RGB to HSI Image")
    st.text("Click this button to convert RGB Image to HSI Image")
    if st.button("RGB to HSI image"):
        plt.figure(figsize=(15,5))
        plt.subplot(1,3,1)
        plt.imshow(HSIimg)
        plt.title("RGB to HSI Image")
        st.pyplot(plt)

    st.subheader("HSI to RGB Image")
    st.text("Click this button to convert HSI Image to RGB Image")
    if st.button("HSI to RGB image"):
        plt.figure(figsize=(15,5))
        plt.subplot(1,3,1)
        plt.imshow(RGBimg)
        plt.title("HSI to RGB Image")
        st.pyplot(plt)
    st.subheader("Histogram Equalization")
    st.text("Click this button to get Equalized RGB image")
    if st.button("RGB image histogram Equalization"):
        heimgrgb = histogram_equalization(rgb_img, "rgb")
        plt.imshow(heimgrgb)
        plt.title("RGB Image Histogram Equalization")
        st.pyplot(plt)
    st.text("Click this button to get Equalized HSI image(converted to RGB)")
    if st.button("HSI image Hsitogram equalized"):
        heimghsi = histogram_equalization(HSIimg, "hsi")
        plt.imshow(heimghsi)
        plt.title("HSI Image Histogram Equalization")
        st.pyplot(plt)
