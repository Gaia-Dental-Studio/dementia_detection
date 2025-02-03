import streamlit as st
import requests

st.title("Dementia Classification")
st.write("Upload an MRI image to classify it into one of the Dementia disease categories.")

# File uploader
uploaded_file = st.file_uploader("Choose an MRI image...", type="jpg")

if uploaded_file is not None:
    # Display uploaded image
    st.image(uploaded_file, caption='Uploaded MRI Image', use_column_width=True)
    
    # Button to trigger prediction
    if st.button("Classify Image"):
        # Send the image to the Flask API
        files = {'file': uploaded_file.getvalue()}
        response = requests.post("http://127.0.0.1:5000/predict", files=files)

        # Display results
        if response.status_code == 200:
            result = response.json()
            st.write(f"**Predicted Class:** {result['predicted_class']}")
            st.write(f"**Confidence:** {result['confidence']:.2f}%")
            st.write("**Confidence scores for each class with threshold 10%:**")
            for class_name, score in result['confidence_scores'].items():
                st.write(f"{class_name}: {score:.2f}%")
        else:
            st.write("Error:", response.json().get("error"))
