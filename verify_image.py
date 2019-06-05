import numpy as np
from PIL import Image

number = int(input("what's the number: "))

img = Image.open('sample%d_r.png'%(number)).convert('L')

img_arr = np.array(img)

#print img_arr.flatten()

WIDTH, HEIGHT = img.size

data = list(img.getdata()) 
data = [data[offset:offset+WIDTH] for offset in range(0, WIDTH*HEIGHT, WIDTH)]


for row in data:
    print(' '.join('{:3}'.format(value) for value in row))
