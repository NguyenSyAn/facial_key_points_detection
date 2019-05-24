import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load a model imported from Tensorflow
tensorflowNet = cv2.dnn.readNetFromTensorflow('output_model_1.pb')

# Input image
image = cv2.imread("face.jpeg", cv2.IMREAD_GRAYSCALE)
image = cv2.resize(image, (96, 96))
flat_image = image.flatten().reshape(96*96)
flat_image = np.vstack(flat_image) / 255
flat_image = np.concatenate(flat_image).ravel()
blob = np.matrix(flat_image)

tensorflowNet.setInput(blob)

# Runs a forward pass to compute the net output
preds = tensorflowNet.forward()

print(preds)

plt.imshow(image, cmap='gray')
plt.scatter(48*preds[0][0::2]+ 48,48*preds[0][1::2]+ 48)
plt.show()