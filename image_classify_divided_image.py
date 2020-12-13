import time
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.preprocessing import image
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import cv2

print("#set classes names")
classes_names = ['animals', 'other', 'person'] #you can change classes

print("#load model")
TF_LITE_MODEL_FILE_NAME = "animall_person_other_v2_fine_tuned.tflite" #you can change model
interpreter = tf.lite.Interpreter(model_path = TF_LITE_MODEL_FILE_NAME)

print("#Check Input Tensor Shape")
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("#Resize Tensor Shape")
interpreter.resize_tensor_input(input_details[0]['index'], (1, 299, 299, 3)) #you can change to your parameters
interpreter.resize_tensor_input(output_details[0]['index'], (1, 3)) #you can change to your parameters
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


#start time
start_time = time.time()

#set image path
img_path = "classify_4_image_divided.jpg" #all classes


####################################BLOCK FOR IMAGE DIVIDING START#############################
# load image for divides
img = cv2.imread(img_path)

##########################################
# At first vertical devide image         #
##########################################
# start vertical devide image
height = img.shape[0]
width = img.shape[1]
# Cut the image in half
width_cutoff = width // 2
left1 = img[:, :width_cutoff]
right1 = img[:, width_cutoff:]
# finish vertical devide image

##########################################
# At first Horizontal devide left1 image #
##########################################
#rotate image LEFT1 to 90 CLOCKWISE
img = cv2.rotate(left1, cv2.ROTATE_90_CLOCKWISE)
# start vertical devide image
height = img.shape[0]
width = img.shape[1]
# Cut the image in half
width_cutoff = width // 2
l2 = img[:, :width_cutoff]
l1 = img[:, width_cutoff:]
# finish vertical devide image
#rotate image to 90 COUNTERCLOCKWISE
l1 = cv2.rotate(l1, cv2.ROTATE_90_COUNTERCLOCKWISE) #image1
#rotate image to 90 COUNTERCLOCKWISE
l2 = cv2.rotate(l2, cv2.ROTATE_90_COUNTERCLOCKWISE) #image2


##########################################
# At first Horizontal devide right1 image#
##########################################
#rotate image RIGHT1 to 90 CLOCKWISE
img = cv2.rotate(right1, cv2.ROTATE_90_CLOCKWISE)
# start vertical devide image
height = img.shape[0]
width = img.shape[1]
# Cut the image in half
width_cutoff = width // 2
r4 = img[:, :width_cutoff]
r3 = img[:, width_cutoff:]
# finish vertical devide image
#rotate image to 90 COUNTERCLOCKWISE
r3 = cv2.rotate(r3, cv2.ROTATE_90_COUNTERCLOCKWISE) #image3
#rotate image to 90 COUNTERCLOCKWISE
r4 = cv2.rotate(r4, cv2.ROTATE_90_COUNTERCLOCKWISE) #image4
####################################BLOCK FOR IMAGE DIVIDING FINISH#############################


#########################################################################################################AT FIRST PREDICT BOTTOM OF IMAGE
##############################################################################################PREDICT R4
# dsize
dsize = (299, 299)
# resize image
img = cv2.resize(r4, dsize) ###NEED CHANGE 
#show image
#plt.imshow(img)
#plt.show()
#image to array
new_img = image.img_to_array(img)
new_img /= 255
new_img = np.expand_dims(new_img, axis=0)
# input_details[0]['index'] = the index which accepts the input
interpreter.set_tensor(input_details[0]['index'], new_img)
# run the inference
interpreter.invoke()   
# The function `get_tensor()` returns a copy of the tensor data.
# Use `tensor()` in order to get a pointer to the tensor.
output_data = interpreter.get_tensor(output_details[0]['index'])
#print(output_data)    
#stop time
elapsed_ms = (time.time() - start_time) * 1000
#print predict classes
classes = np.argmax(output_data, axis = 1)
print("IMAGE R4. Elapsed time: ", elapsed_ms, " , predict class number: ", classes, " ,is class name: ", classes_names[classes[0]], sep='')
##############################################################################################PREDICT L2
# dsize
dsize = (299, 299)
# resize image
img = cv2.resize(l2, dsize) ###NEED CHANGE 
#show image
#plt.imshow(img)
#plt.show()
#image to array
new_img = image.img_to_array(img)
new_img /= 255
new_img = np.expand_dims(new_img, axis=0)
# input_details[0]['index'] = the index which accepts the input
interpreter.set_tensor(input_details[0]['index'], new_img)
# run the inference
interpreter.invoke()   
# The function `get_tensor()` returns a copy of the tensor data.
# Use `tensor()` in order to get a pointer to the tensor.
output_data = interpreter.get_tensor(output_details[0]['index'])
#print(output_data)    
#stop time
elapsed_ms = (time.time() - start_time) * 1000
#print predict classes
classes = np.argmax(output_data, axis = 1)
print("IMAGE L2. Elapsed time: ", elapsed_ms, " , predict class number: ", classes, " ,is class name: ", classes_names[classes[0]], sep='')


#########################################################################################################AT FIRST PREDICT TOP OF IMAGE
##############################################################################################PREDICT L1
# dsize
dsize = (299, 299)
# resize image
img = cv2.resize(l1, dsize) ###NEED CHANGE 
#show image
#plt.imshow(img)
#plt.show()
#image to array
new_img = image.img_to_array(img)
new_img /= 255
new_img = np.expand_dims(new_img, axis=0)
# input_details[0]['index'] = the index which accepts the input
interpreter.set_tensor(input_details[0]['index'], new_img)
# run the inference
interpreter.invoke()   
# The function `get_tensor()` returns a copy of the tensor data.
# Use `tensor()` in order to get a pointer to the tensor.
output_data = interpreter.get_tensor(output_details[0]['index'])
#print(output_data)    
#stop time
elapsed_ms = (time.time() - start_time) * 1000
#print predict classes
classes = np.argmax(output_data, axis = 1)
print("IMAGE L1. Elapsed time: ", elapsed_ms, " , predict class number: ", classes, " ,is class name: ", classes_names[classes[0]], sep='')
##############################################################################################PREDICT R3
# dsize
dsize = (299, 299)
# resize image
img = cv2.resize(r3, dsize) ###NEED CHANGE 
#show image
#plt.imshow(img)
#plt.show()
#image to array
new_img = image.img_to_array(img)
new_img /= 255
new_img = np.expand_dims(new_img, axis=0)
# input_details[0]['index'] = the index which accepts the input
interpreter.set_tensor(input_details[0]['index'], new_img)
# run the inference
interpreter.invoke()   
# The function `get_tensor()` returns a copy of the tensor data.
# Use `tensor()` in order to get a pointer to the tensor.
output_data = interpreter.get_tensor(output_details[0]['index'])
#print(output_data)    
#stop time
elapsed_ms = (time.time() - start_time) * 1000
#print predict classes
classes = np.argmax(output_data, axis = 1)
print("IMAGE R3. Elapsed time: ", elapsed_ms, " , predict class number: ", classes, " ,is class name: ", classes_names[classes[0]], sep='')

