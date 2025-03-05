import tensorflow as tf
from tensorflow import keras


def make_model(input_shape, hidden_size, output_size):


    state_input = keras.layers.Input(shape=input_shape,dtype =float)

    layer1 = keras.layers.Dense(hidden_size, activation="relu")(state_input)
    layer2 = keras.layers.Dropout(0.5)(layer1)
    layer3 = keras.layers.Dense(hidden_size, activation="relu")(layer2)
    layer4 = keras.layers.Dropout(0.5)(layer3)




    image_input = keras.layers.Input(shape = (28,28,1),dtype =float)
    conv2D = keras.layers.Conv2D(32, (3,3), activation="relu")(image_input)
    pooling = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2D)
    flatten = keras.layers.Flatten()(pooling)
    imageLayer1= keras.layers.Dense(hidden_size, activation="relu")(flatten)
    imageLayer2 = keras.layers.Dropout(0.5)(imageLayer1)
    imageLayer3 = keras.layers.Dense(hidden_size, activation="relu")(imageLayer2)
    imageLayer4 = keras.layers.Dropout(0.5)(imageLayer3)


    concatetenated = keras.layers.Concatenate()([layer4,imageLayer4])

    output = keras.layers.Dense(output_size, activation="linear")(concatetenated)

    model = keras.models.Model([state_input,image_input],output)


    model = keras.models.load_model(r"C:\Users\admin\Desktop\ORGANISED\CODE\models\snake.keras")



    return model
