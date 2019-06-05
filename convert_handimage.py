import numpy as np
from PIL import Image
import pandas as pd

#for i in range(0,10,1):

number = int(input("what's the number: "))

img = Image.open('sample%d_black_r.png'%(number)).convert('L')

print np.array(img)
img_arr = np.array(img)

print "hello hello"

print img_arr.flatten()

flat_img_arr = img_arr.flatten()

print "check whether understanding of flattening works or not: "

print img_arr[3,15], flat_img_arr[99] # works quite well 

img_list = flat_img_arr.tolist()

print type(img_list)
print img_list[99], "length of the list: ", len(img_list)

print "length of the image list bfore: ", len(img_list)

# add the number in the list of pixels as a form of label
print number
img_list.insert(0, number)

print "length of the image list after: ", len(img_list)

print img_list


MNIST_test_small_df = pd.read_csv('mnist_test.csv')
#print MNIST_test_small_df.head(3)
#print type(MNIST_test_small_df.columns)
column_list = list(MNIST_test_small_df.columns.values)
print ("length of column list",len(column_list))

hand_df = pd.DataFrame([img_list], columns=column_list)

hand_df.to_csv('hand_check.csv', encoding='utf-8', )

print hand_df.head(2)
