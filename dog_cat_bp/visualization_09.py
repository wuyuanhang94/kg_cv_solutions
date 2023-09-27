from keras.models import load_model

model = load_model('cats_and_dogs_small_2.h5')
print(model.summary())

img_path = '/home/yi/daily/red_peo_code/dog_cat/small/test/cats/cat.1700.jpg'

from keras.preprocessing import image
import numpy as np

img = image.load_img(img_path, target_size=(150, 150))
img_tensor = image.img_to_array(img)
img_tensor = np.expand_dims(img_tensor, axis=0)
img_tensor /= 255.
# 如果没有归一化 使用 plt.imshow(img_tensor[0].astype('uint8'))

print(img_tensor)

import matplotlib.pyplot as plt

# plt.imshow(img_tensor[0])   
# plt.show()

from keras import models

layer_outputs = [layer.output for layer in model.layers[:8]]
activation_model = models.Model(inputs=model.input, outputs=layer_outputs)

activations = activation_model.predict(img_tensor)

# plt.matshow(activations[0][0, :, :, 7], cmap='viridis')
# plt.matshow(activations[0][0, :, :, 4], cmap='viridis')
# plt.show()

layer_names = []
for layer in model.layers[:8]:
    layer_names.append(layer.name)

imgs_per_row = 16

for layer_name, layer_activation in zip(layer_names, activations):
    n_features = layer_activation.shape[-1]
    