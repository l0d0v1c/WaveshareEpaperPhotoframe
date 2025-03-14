# Photo Formatter for Waveshare Phoyoframe ePaper

This Python script is specifically designed to format photos for display on the **Waveshare Phoframe ePaper**. It processes input images by resizing and generating appropriately cropped versions to meet the device's display requirements.

## Features

- **Horizontal Resize:**  
  The script first resizes the input image proportionally so that its width is exactly **800 pixels**.

- **Vertical Cropping:**  
  - **If the resized image's height is greater than or equal to 480 pixels:**  
    The script generates multiple crops (by sliding a vertical window) of **800×480 pixels**.  
    Crops are saved with suffixes such as `top`, `upper`, `lower`, and `bottom` to indicate their vertical position.
  - **Additionally,** in this case, a "forced" resized image is also generated by resizing (without preserving the aspect ratio) the image to **800×480 pixels**. This file is saved with the suffix `_resized`.

- **Forced Resize:**  
  If the height of the proportionally resized image is less than 480 pixels, the image is forcibly resized (which may distort the image) to **800×480 pixels**.

- **Quantization with Dithering:**  
  The script applies Floyd–Steinberg dithering using a fixed color palette contained in the file `N-color.act`.

- **Optimized Output:**  
  All generated images are saved in **24-bit BMP** format, which is required for the Waveshare Phoframe ePaper.

## Prerequisites

- **Python 3.x**

- **Pillow** (the Python Imaging Library)  
  Install it via pip:
  ```bash
  pip install Pillow
  	•	The palette file N-color.act must be located in the same directory as the script.
  	
## Usage

Run the script from the command line:

`python script.py input_image [output_image]`
