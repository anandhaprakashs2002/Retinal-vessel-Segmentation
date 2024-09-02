import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np

# Define the custom layer class
class FixedDropout(tf.keras.layers.Layer):
    def __init__(self, rate, noise_shape=None, seed=None, **kwargs):
        super(FixedDropout, self).__init__(**kwargs)
        self.rate = rate
        self.noise_shape = noise_shape
        self.seed = seed

    def call(self, inputs, training=None):
        if self.noise_shape is None:
            noise_shape = None
        else:
            noise_shape = tf.shape(inputs)
            noise_shape = [noise_shape[i] if shape is None else shape for i, shape in enumerate(self.noise_shape)]

        return tf.nn.dropout(inputs, rate=self.rate, noise_shape=noise_shape, seed=self.seed)

# Define preprocess_input function for image preprocessing
def preprocess_input(image):
    # Resize the image to match the input size expected by the model
    image = image.resize((512, 512))
    # Normalize the pixel values to the range [0, 1]
    image = np.array(image) / 255.0
    return image

# Load the trained segmentation model
model_path = "for_mod.h5"  # Update with your model path
with tf.keras.utils.custom_object_scope({'FixedDropout': FixedDropout}):
    model = tf.keras.models.load_model(model_path)

# Function to perform segmentation inference
def segment_image(input_image):
    # Preprocess input image
    input_image = preprocess_input(input_image)
    
    # Perform segmentation inference
    segmented_image = model.predict(np.expand_dims(input_image, axis=0))[0]
    
    # Post-process segmented image if needed
    # (e.g., thresholding, resizing, converting to uint8)
    
    return segmented_image

# Streamlit UI
st.title("Welcome to Retina Vessel Segmentation ")
st.sidebar.title(" Segmentated Image ")

uploaded_image = st.sidebar.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_image is not None:
    # Display uploaded image
    st.sidebar.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
    
    # Perform segmentation inference
    segmented_image = segment_image(Image.open(uploaded_image))
    
    # Display segmented image
    st.image(segmented_image, caption="Segmented Image", use_column_width=True)
    
    # Add additional information about retinal vessel diseases
    st.write("""
    - **Retinal Vessel Diseases**: These include diseases affecting the blood vessels in the retina, such as diabetic retinopathy, hypertensive retinopathy, and retinal vein occlusion.
    - **Implications**: Damage to retinal vessels can lead to vision loss or impairment. For example, in diabetic retinopathy, abnormal blood vessel growth can cause bleeding and scarring, leading to vision loss.
    - **Detection and Treatment**: Early detection through screening is crucial for managing retinal vessel diseases. Treatments may include laser therapy, medication injections, or surgery, depending on the specific condition.
    - **Risk Factors**: Risk factors for retinal vessel diseases include diabetes, hypertension, high cholesterol, smoking, and a family history of eye diseases.
    """)
