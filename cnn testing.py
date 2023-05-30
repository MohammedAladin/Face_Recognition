import joblib
import numpy as np
from Knn import KNN
from FaceReco import *
import tensorflow as tf

trPath = 'Train'
tePath = 'Test'

if __name__ == "__main__":
    my_map = dict([(1, 'Som3a'), (2, 'Sayed'), (3, 'Abdo'), (4, 'Omar'), (5, 'Alaa')])

    interpreter = tf.lite.Interpreter(model_path='my_model.tflite')
    interpreter.allocate_tensors()

    # Get input and output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Load and preprocess the test image
    my_image, _ = findEcoding("210835517_2992373561025337_2594230193498265018_n.jpg")
    my_image = tf.keras.preprocessing.image.img_to_array(my_image)
    my_image = my_image.reshape(1, 128, 1)
    input_data = np.array(my_image, dtype=np.float32)

    # Set the input tensor
    interpreter.set_tensor(input_details[0]['index'], input_data)

    # Run inference
    interpreter.invoke()

    # Get the output tensor
    output_data = interpreter.get_tensor(output_details[0]['index'])

    # Print the labels
    for i in range(len(output_data)):
        print("Pridected Name:",my_map.get(output_data[i].argmax()))
 