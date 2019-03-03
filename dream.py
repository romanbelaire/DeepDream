#DEEP DREAM PROJECT
#ROMAN BELAIRE

import numpy as np
import scipy
import PIL.Image
import os
import h5py
import argparse
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.image import load_img, save_img, img_to_array
from keras.applications import inception_v3
from keras import backend as K

#argument parser
#parser = argparse.ArgumentParser()
#parser.add_argument("-r", "--retrain", help="Retrain the model.", action="store_true")
#parser.add_argument("-d", "--dataset_directory", help="Directory containing dataset. Should be sorted into [directory]/train, [directory]/validate, [directory]/test.",
#                    default="resources/dataset/")


# To fix FailedPreconditionError:
sess = tf.InteractiveSession()
#show connected devices:
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

with tf.Session() as sess:
         sess.run(tf.global_variables_initializer())

print("GPUs: ")
K.tensorflow_backend._get_available_gpus()
######## The following code is based on the keras documentation opposite
#aimed at making my life easier and creating my retrained inception model without raw tensorflow
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras import Sequential
from keras.layers import Dense, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model

#constant vars
NUM_CLASSES = 9
batchsize = 32
EPOCHS = 500
#from keras import backend as K

def retrain_model():
# create the base pre-trained model
    base_model = InceptionV3(weights='imagenet', include_top=False)

    # add a global spatial average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # let's add a fully-connected layer
    x = Dense(1024, activation='relu')(x)
    # and a logistic layer -- let's say we have 200 classes
    predictions = Dense(NUM_CLASSES, activation='softmax')(x) #had to make sure the number of classes matched up. fuckin keras doc hard-coded 200 classes

    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)

    # first: train only the top layers (which were randomly initialized)
    # i.e. freeze all convolutional InceptionV3 layers
    for layer in base_model.layers:
        layer.trainable = False

    # compile the model (should be done *after* setting layers to non-trainable)
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

    data_aug = ImageDataGenerator(rotation_range=20, zoom_range=0.15,
    	width_shift_range=0.2, height_shift_range=0.2, shear_range=0.15,
    	horizontal_flip=True, fill_mode="nearest")

    # train the model on the new data for a few epochs
    #data generators
    train_gen = data_aug.flow_from_directory(directory = 'resources/dataset/train',
                                            target_size = (255, 255), color_mode='rgb',
                                            batch_size=batchsize, class_mode='categorical',
                                            shuffle='True', seed=420) #its important that the seed is an int and not a string lol
    val_gen = data_aug.flow_from_directory(directory = 'resources/dataset/validate',
                                            target_size = (255, 255), color_mode='rgb',
                                            batch_size=batchsize, class_mode='categorical',
                                            shuffle='True', seed=69)
    print("data generators loaded.")

    STEP_SIZE_TRAIN=train_gen.n//train_gen.batch_size
    STEP_SIZE_VALID=val_gen.n//val_gen.batch_size

    model.fit_generator(generator=train_gen,
                        steps_per_epoch=STEP_SIZE_TRAIN,
                        validation_data=val_gen,
                        validation_steps=STEP_SIZE_VALID,
                        epochs=EPOCHS)
    # at this point, the top layers are well trained and we can start fine-tuning
    # convolutional layers from inception V3. We will freeze the bottom N layers
    # and train the remaining top layers.

    # let's visualize layer names and layer indices to see how many layers
    # we should freeze:
    for i, layer in enumerate(base_model.layers):
       print(i, layer.name)

    # we chose to train the top 2 inception blocks, i.e. we will freeze
    # the first 249 layers and unfreeze the rest:
    for layer in model.layers[:249]:
       layer.trainable = False
    for layer in model.layers[249:]:
       layer.trainable = True

    # we need to recompile the model for these modifications to take effect
    # we use SGD with a low learning rate
    from keras.optimizers import SGD
    model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy')

    # we train our model again (this time fine-tuning the top 2 inception blocks
    # alongside the top Dense layers
    model.fit_generator(generator=train_gen,
                        steps_per_epoch=STEP_SIZE_TRAIN,
                        validation_data=val_gen,
                        validation_steps=STEP_SIZE_VALID,
                        epochs=EPOCHS)

    print("finished generator successfully")
    #save our stuff
    model_json = model.to_json()
    with open("resources/saved_models/model1/model.json", "w") as json_file:
        json_file.write(model_json)
    model.save_weights("resources/saved_models/model1/model_weights.h5")
    print("weights saved.")

    model.save("resources/saved_models/model1/full_model.h5")
    print("full model saved")

    return model
    ########end


def load_full_model(path):
    return load_model(path)


# Disable all training specific operations
K.set_learning_phase(0)

# The model will be loaded with pre-trained inceptionv3 weights.
#JK WE USIN MY BRAND NEW SHARK TRAINED MODEL
#model = inception_v3.InceptionV3(weights='resources/output_model.h5', include_top=False)
#model = load_full_model("resources/saved_models/model1/full_model.h5")
model = retrain_model()
dream = model.input
print('Model loaded.')


# You can tweak these setting to obtain new visual effects.
settings = {
    'features': {
        'mixed2': 2, #wavy layers
        'mixed3': 1.5, #smooth circles
        'mixed4': 6,  #kind of jagged
        'mixed5': 0.5,    #wrinkle/fur texture
    },
}


# Set a function to load, resize and convert the image.
def preprocess_image(image_path):
    # Util function to open, resize and format pictures
    # into appropriate tensors.
    img = load_img(image_path)
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = inception_v3.preprocess_input(img)
    return img


# And a function to do the opposite: convert a tensor into an image.
def deprocess_image(x):
    # Util function to convert a tensor into a valid image.
    if K.image_data_format() == 'channels_first':
        x = x.reshape((3, x.shape[2], x.shape[3]))
        x = x.transpose((1, 2, 0))
    else:
        x = x.reshape((x.shape[1], x.shape[2], 3))
    x /= 2.
    x += 0.5
    x *= 255.
    x = np.clip(x, 0, 255).astype('uint8')
    return x


# Set a dictionary that maps the layer name to the layer instance.
# get the symbolic outputs of each "key" layer (we gave them unique names).
layer_dict = dict([(layer.name, layer) for layer in model.layers])


# Define the loss. The way this works is first the scalar variable *loss* is set.
# Then the loss will be defined by adding layer contributions to this variable.
loss = K.variable(0.)

for layer_name in settings['features']:
    # Add the L2 norm of the features of a layer to the loss.
    assert (layer_name in layer_dict.keys(),
            'Layer ' + layer_name + ' not found in model.')
    coeff = settings['features'][layer_name]
    x = layer_dict[layer_name].output
    # We avoid border artifacts by only involving non-border pixels in the loss.
    scaling = K.prod(K.cast(K.shape(x), 'float32'))
    if K.image_data_format() == 'channels_first':
        loss += coeff * K.sum(K.square(x[:, :, 2: -2, 2: -2])) / scaling
    else:
        loss += coeff * K.sum(K.square(x[:, 2: -2, 2: -2, :])) / scaling


# Compute the gradients of the dream wrt the loss.
grads = K.gradients(loss, dream)[0]
# Normalize gradients.
grads /= K.maximum(K.mean(K.abs(grads)), K.epsilon())

# Set up function to retrieve the value of the loss and gradients given an input image.
outputs = [loss, grads]
fetch_loss_and_grads = K.function([dream], outputs)

def eval_loss_and_grads(x):
    outs = fetch_loss_and_grads([x])
    loss_value = outs[0]
    grad_values = outs[1]
    return loss_value, grad_values

# Helper funtion to resize
def resize_img(img, size):
    img = np.copy(img)
    if K.image_data_format() == 'channels_first':
        factors = (1, 1,
                   float(size[0]) / img.shape[2],
                   float(size[1]) / img.shape[3])
    else:
        factors = (1,
                   float(size[0]) / img.shape[1],
                   float(size[1]) / img.shape[2],
                   1)
    return scipy.ndimage.zoom(img, factors, order=1)


# Define the gradient ascent function over a number of iterations.
def gradient_ascent(x, iterations, step, max_loss=None):
    for i in range(iterations):
        loss_value, grad_values = eval_loss_and_grads(x)
        if max_loss is not None and loss_value > max_loss:
            break
        print('..Loss value at', i, ':', loss_value)
        x += step * grad_values
    return x


# Set hyperparameters. The ocatave_scale is the ratio between each successive scale (remember the upscaling mentioned before?).
# Playing with these hyperparameters will also allow you to achieve new effects
step = 0.008  # Gradient ascent step size
num_octave = 5  # Number of scales at which to run gradient ascent
octave_scale = 1.4  # Size ratio between scales
iterations = 20  # Number of ascent steps per scale
max_loss = 10.

base_image_path = "resources/images/ducks_0.jpg"
print('opening ' + base_image_path)
img = PIL.Image.open(base_image_path)
img

img = preprocess_image(base_image_path)
if K.image_data_format() == 'channels_first':
    original_shape = img.shape[2:]
else:
    original_shape = img.shape[1:3]
successive_shapes = [original_shape]
for i in range(1, num_octave):
    shape = tuple([int(dim / (octave_scale ** i)) for dim in original_shape])
    successive_shapes.append(shape)
successive_shapes = successive_shapes[::-1]
original_img = np.copy(img)
shrunk_original_img = resize_img(img, successive_shapes[0])

for shape in successive_shapes:
    print('Processing image shape', shape)
    img = resize_img(img, shape)
    img = gradient_ascent(img,
                          iterations=iterations,
                          step=step,
                          max_loss=max_loss)
    upscaled_shrunk_original_img = resize_img(shrunk_original_img, shape)
    same_size_original = resize_img(original_img, shape)
    lost_detail = same_size_original - upscaled_shrunk_original_img

    img += lost_detail
    shrunk_original_img = resize_img(original_img, shape)

save_img('dream.jpg',deprocess_image(np.copy(img)))
print('saved dream')
dreamout = PIL.Image.open('dream.jpg')
dreamout
