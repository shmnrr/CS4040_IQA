from PIL import Image
import pandas as pd
import os
import time

def jpg2png_resize(n: int, size: int=512):
    start_time = time.time()
    count = 0
    for filename in os.listdir("./datasets/SPAQ/TestImage"):
        if count >= n:
            break
        if filename.endswith(".jpg"):
            img = Image.open("./datasets/SPAQ/TestImage/" + filename)
            width, height = img.size
            if width < height:
                new_width = size
                new_height = int(height * new_width / width)
            else:
                new_height = size
                new_width = int(width * new_height / height)
            
            resized_img = img.resize((new_width, new_height), Image.LANCZOS)
            resized_img.save("./datasets/SPAQ/TestImage/" + filename[:-4] + ".png", "PNG")
            os.remove("./datasets/SPAQ/TestImage/" + filename)
            count += 1
            if count % 100 == 0:
                print(f"Converted {count} images in {time.time() - start_time} seconds")
        else:
            continue

    print(f"Converted {count} images in {time.time() - start_time} seconds")

if __name__ == '__main__':
    NUM_IMAGES = 1000
    print(f"Converting {NUM_IMAGES} images from jpg to png with shorter side of 512 pixels")
    jpg2png_resize(NUM_IMAGES)