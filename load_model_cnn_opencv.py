import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load a model imported from Tensorflow
tensorflowNet = cv2.dnn.readNetFromTensorflow('output_model_3.pb')

# Input image
image = cv2.imread("face.jpeg", cv2.IMREAD_GRAYSCALE)
image = cv2.resize(image, (96, 96))

# Use the given image as input, which needs to be blob(s).
inScaleFactor = 1 / 255
blob = cv2.dnn.blobFromImage(image, inScaleFactor, (96, 96), ddepth = cv2.CV_32F)
print(blob.shape)
tensorflowNet.setInput(blob)

# Runs a forward pass to compute the net output
preds = tensorflowNet.forward()

plt.imshow(image, cmap='gray')
plt.scatter(48*preds[0][0::2]+ 48,48*preds[0][1::2]+ 48)
plt.show()