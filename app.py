import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import random
from PIL import Image

# 🔹 Ensure set_page_config() is the FIRST Streamlit command
st.set_page_config(page_title="Mood-Based Song Recommender", page_icon="🎵", layout="centered")

# Load the trained model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('mood_classification_tuned.h5')

model = load_model()

# Load dataset (CSV) containing songs and moods
csv_path = 'data_moods.csv'  # Ensure correct path
df = pd.read_csv(csv_path)
df['mood'] = df['mood'].str.lower()  # Ensure consistency

# Preprocess an image for prediction
def preprocess_image(img, img_size=(128, 128)):
    img = img.resize(img_size)  # Resize image
    img = img.convert("RGB")  # 🔹 Convert grayscale to RGB
    img_array = image.img_to_array(img)  # Convert to NumPy array
    img_array = np.expand_dims(img_array, axis=0)  # Expand dimensions
    img_array /= 255.0  # Normalize
    return img_array

# Predict mood
def predict_mood(img, class_indices):
    img_array = preprocess_image(img)
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction, axis=1)[0]
    class_labels = {v: k for k, v in class_indices.items()}  # Reverse mapping
    return class_labels[predicted_class], prediction[0][predicted_class]

# Recommend a song based on the predicted mood
def recommend_song(mood):
    mood_songs = df[df['mood'] == mood]['song'].tolist()
    return random.choice(mood_songs) if mood_songs else "No matching song found for this mood."

# Example class indices (Replace with actual mapping)
class_indices = {'calm': 0, 'energetic': 1, 'happy': 2, 'sad': 3}

# Streamlit UI
def main():
    st.title("🎵 Mood-Based Song Recommender 🎶")
    st.write("Upload an image, and we'll analyze your mood and recommend the perfect song!")

    uploaded_file = st.file_uploader("Upload a .jpg image", type=["jpg", "jpeg"])

    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)  # 🔹 Fixed Warning

        img = Image.open(uploaded_file)

        # Predict mood
        predicted_mood, confidence = predict_mood(img, class_indices)

        # Recommend a song
        recommended_song = recommend_song(predicted_mood)

        # Display results
        st.success(f"**Predicted Mood:** {predicted_mood.capitalize()} (Confidence: {confidence:.2f})")
        st.subheader(f"🎶 This song perfectly suits your mood: **{recommended_song}**")

if __name__ == "__main__":
    main()
