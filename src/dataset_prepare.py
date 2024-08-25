import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import os

# Convert string to integer
def atoi(s):
    n = 0
    for i in s:
        n = n * 10 + ord(i) - ord("0")
    return n

# Define paths and make directories
base_dir = 'data'
outer_names = ['test', 'train']
inner_names = ['angry', 'disgusted', 'fearful', 'happy', 'sad', 'surprised', 'neutral']

# Create directories if they don't exist
for outer_name in outer_names:
    for inner_name in inner_names:
        dir_path = os.path.join(base_dir, outer_name, inner_name)
        os.makedirs(dir_path, exist_ok=True)

# Initialize counters
counts = {name: 0 for name in inner_names}
test_counts = {name + '_test': 0 for name in inner_names}

# Read the CSV file
df = pd.read_csv('.\\fer2013.csv\\fer2013.csv')
mat = np.zeros((48, 48), dtype=np.uint8)
print("Saving images...")

# Process each row in the CSV file
for i in tqdm(range(len(df))):
    txt = df['pixels'][i]
    words = txt.split()
    
    # Convert string pixels to numpy array
    for j in range(2304):
        xind = j // 48
        yind = j % 48
        mat[xind][yind] = atoi(words[j])
        
    img = Image.fromarray(mat)

    # Determine the directory and save the image
    emotion = df['emotion'][i]
    emotion_name = inner_names[emotion]
    if i < 28709:  # Training data
        save_dir = os.path.join(base_dir, 'train', emotion_name)
        img.save(os.path.join(save_dir, f'im{counts[emotion_name]}.png'))
        counts[emotion_name] += 1
    else:  # Testing data
        save_dir = os.path.join(base_dir, 'test', emotion_name)
        img.save(os.path.join(save_dir, f'im{test_counts[emotion_name + "_test"]}.png'))
        test_counts[emotion_name + '_test'] += 1

print("Done!")
