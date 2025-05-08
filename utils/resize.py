import os
from PIL import Image

# Path to the input folder containing images
input_folder = "utils/ImageGen2056"
# Path to the output folder where resized images will be saved
output_folder = "utils/dcGAN_resized"

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Set the new size (width, height)
new_size = (546, 199)

def resize_images_in_folder(input_folder, output_folder, new_size):
    for root, _, files in os.walk(input_folder):
        for filename in files:
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):  # Filter image files
                input_path = os.path.join(root, filename)  # Full input path
                
                # Create corresponding output path by preserving folder structure
                relative_path = os.path.relpath(input_path, input_folder)
                output_path = os.path.join(output_folder, relative_path)
                
                # Create target folder if it doesn't exist
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                
                try:
                    # Open and resize image
                    with Image.open(input_path) as img:
                        img_resized = img.resize(new_size, Image.ANTIALIAS)
                        # Save resized image to output path
                        img_resized.save(output_path)
                        #print(f"Resized and saved: {output_path}")
                except Exception as e:
                    print(f"Failed to resize {input_path}: {e}")

# Run the resizing function
resize_images_in_folder(input_folder, output_folder, new_size)

print("All images resized successfully!")