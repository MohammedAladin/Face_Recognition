import numpy as np
import joblib
import numpy as np
from FaceReco import *
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder

trPath = 'Train'
tePath = 'Test'


if __name__ == "__main__":
    
    my_map = dict([(1, 'Som3a'), (2, 'Sayed'),(3, 'Abdo'), (4, 'Omar'),(5, 'Alaa') ])

    train_data,train_label=load_data(trPath)
    test_data,test_label = load_data(tePath)

    train_data = np.array(train_data)
    test_data = np.array(test_data)

    train_label = np.array(train_label)
    train_label = train_label.astype(int)

    test_label = np.array(test_label)
    test_label = test_label.astype(int)
    

    train_data = train_data.reshape(-1, 128, 1)
    test_data = test_data.reshape(-1, 128, 1)

    model = tf.keras.Sequential([
    tf.keras.layers.Conv1D(32, 3, activation='relu', input_shape=(128, 1)),
    tf.keras.layers.MaxPooling1D(2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(6, activation='softmax')
])

# Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(train_data, train_label, epochs=100)

    # Evaluate the model
    _, accuracy = model.evaluate(test_data, test_label)
    print('Test accuracy:', 100*accuracy)

    predictions = model.predict(test_data)

    # Print the labels
    for i in range(len(predictions)):
         print(my_map.get(predictions[i].argmax()))

    model.save('my_model.h5')
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    # Save the TFLite model
    with open('my_model.tflite', 'wb') as f:
        f.write(tflite_model)



        


    #camModel(train_data,train_label) 
     
