from PIL import Image
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def classify_uploaded_image(model_path, img_path):
    model = tf.keras.models.load_model(model_path)
    img = Image.open(img_path).convert('RGB')
    img = img.resize((150, 150))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    if prediction[0] > 0.5:
        result = "Non-Healthy"
        confidence = prediction[0][0] * 100
    else:
        result = "Healthy"
        confidence = (1 - prediction[0][0]) * 100
    plt.imshow(img)
    plt.axis('off')
    plt.title(f"Predicted: {result} ({confidence:.2f}% confidence)")
    plt.show()
    print(f"The uploaded lettuce is classified as: {result}")
    print(f"Confidence score: {confidence:.2f}%")

if __name__ == "__main__":
    classify_uploaded_image("healthy_vs_non_healthy_classifier.keras", r"C:\Users\jazmi\Downloads\Rhizoctonia-Stem-Rot-in-Lettuce.jpeg")
