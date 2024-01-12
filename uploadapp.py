import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
def predict(image):
    # print (image.size)
    resized_image = image.resize((32, 32))
    # print (resized_image.size)
    img_array = np.array(resized_image.convert('RGB'))
    normalized_image = img_array / 255.0
    model = load_model("achawala.h5")
    ans = model.predict(normalized_image.reshape(1, 32, 32, 3))
    if ans[0][0]<ans[0][1]:
        return "Non-Anemic"
    else:
        return "Anemic"
    return ans
def main():
    st.title("Image Prediction App")

    uploaded_file = st.file_uploader("Choose a file", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        if st.button("Predict"):
            prediction_result = predict(image)
            del image
            st.write("Prediction Result:", prediction_result)

if __name__ == "__main__":
    main()
