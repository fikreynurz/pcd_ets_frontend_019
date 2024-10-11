import io
import requests
import streamlit as st
import cv2
import os
import numpy as np
import time
import uuid
from skimage.exposure import match_histograms
import matplotlib.pyplot as plt
from PIL import Image  # Import PIL for image handling

# URLs for FastAPI backend
BACKEND_URL = "http://localhost:8000"
ENDPOINT_ADD_NOISE = f"{BACKEND_URL}/add_noise"
ENDPOINT_REMOVE_NOISE = f"{BACKEND_URL}/remove_noise"
ENDPOINT_SHARPEN_IMAGE = f"{BACKEND_URL}/sharpen_image"
ENDPOINT_RGB_EXTRACTION = f"{BACKEND_URL}/rgb_extraction"
ENDPOINT_OPERATION_RGB = f"{BACKEND_URL}/operation_rgb"
ENDPOINT_LOGIC_OPERATION_RGB = f"{BACKEND_URL}/logic_operation_rgb"
ENDPOINT_GRAYSCALE = f"{BACKEND_URL}/grayscale"
ENDPOINT_HISTOGRAM = f"{BACKEND_URL}/histogram"
ENDPOINT_EQUALIZE = f"{BACKEND_URL}/equalize"
ENDPOINT_SPECIFY_HISTOGRAM = f"{BACKEND_URL}/specify_histogram"
ENDPOINT_STATISTICS = f"{BACKEND_URL}/statistics"

# Ensure necessary directories exist
if not os.path.exists("processed_dataset"):
    os.makedirs("processed_dataset")

# Sidebar Navigation
st.sidebar.title("Navigation")
app_mode = st.sidebar.selectbox("Choose the app mode",
                                ["Face Detection & Image Processing", "RGB Extraction & Image Operations"])

# Display User Identity
st.markdown("### **Name:** Muhammad Fikri Nur Sya'Bani")
st.markdown("### **NIM:** 221524019")
st.markdown("### **Class:** D4 - 3A")
st.markdown("---")

# ---------------------- Face Detection and Image Processing Section ---------------------- #
if app_mode == "Face Detection & Image Processing":
    st.title("Face Detection and Image Processing")

    # Input for new person
    new_person = st.text_input("Enter the name of the new person:")

    # Button to capture new face
    capture = st.button("Add New Face")

    if capture:
        if not new_person:
            st.warning("Please enter the name of the new person.")
        else:
            save_path = os.path.join('dataset', new_person)
            os.makedirs(save_path, exist_ok=True)
            st.success(f"Folder for {new_person} has been created.")

            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                st.error("Cannot open webcam. Please ensure the webcam is connected and not used by another application.")
            else:
                num_images = 0
                max_images = 20

                frame_placeholder = st.empty()
                progress_bar = st.progress(0)
                status_text = st.empty()

                try:
                    while num_images < max_images:
                        ret, frame = cap.read()
                        if not ret:
                            st.error("Error: Cannot read frame from webcam.")
                            break

                        # Convert frame to RGB for display
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        frame_placeholder.image(frame_rgb, channels="RGB", caption="Capturing Faces...")

                        # Detect faces using OpenCV
                        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

                        if len(faces) > 0:
                            for (x, y, w, h) in faces:
                                face = frame[y:y+h, x:x+w]
                                unique_id = uuid.uuid4().hex
                                img_name = os.path.join(save_path, f"img_{unique_id}.jpg")
                                cv2.imwrite(img_name, face)
                                num_images += 1

                                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                                frame_placeholder.image(frame_rgb, channels="RGB", caption=f"Image {num_images}/{max_images}")

                                progress = num_images / max_images
                                progress_bar.progress(progress)
                                status_text.text(f"Saving image {num_images} of {max_images}...")
                                break  # Save one face per frame
                        else:
                            frame_placeholder.image(frame_rgb, channels="RGB", caption="No face detected.")

                        time.sleep(0.1)

                    st.success(f"{num_images} images have been successfully added to the {new_person} dataset.")
                finally:
                    cap.release()
                    frame_placeholder.empty()
                    progress_bar.empty()
                    status_text.empty()

    st.header("Image Processing Options")

    image_width = st.slider("Adjust image width:",
                            min_value=100, max_value=1000, value=200)

    salt_color = st.color_picker('Select Salt Color', '#FFFFFF')
    pepper_color = st.color_picker('Select Pepper Color', '#000000')

    dataset_folder = 'dataset'
    if os.path.exists(dataset_folder):
        all_images = []
        for root, dirs, files in os.walk(dataset_folder):
            for file in files:
                if file.endswith(('.png', '.jpg', '.jpeg')):
                    all_images.append(os.path.join(root, file))

        if all_images:
            selected_image_path = st.selectbox(
                "Select an image to process:", all_images)

            if selected_image_path:
                image = cv2.imread(selected_image_path)
                st.image(image, caption="Selected Image Preview",
                         channels="BGR", width=image_width)

            noise_probability = st.slider(
                "Set Salt & Pepper Noise Level:", min_value=0.0, max_value=1.0, value=0.05)

            process = st.button("Process Image")

            if process:
                if selected_image_path:
                    if noise_probability < 0 or noise_probability > 1:
                        st.error("Noise level must be between 0.0 and 1.0.")
                    else:
                        with st.spinner('Processing image...'):
                            # Send Add Noise Request
                            with open(selected_image_path, "rb") as f:
                                files = {
                                    "file": ("image.jpg", f, "image/jpeg")
                                }
                                data = {
                                    "noise_prob": noise_probability,
                                    "salt_color": salt_color,
                                    "pepper_color": pepper_color
                                }
                                response = requests.post(ENDPOINT_ADD_NOISE, files=files, data=data)
                            
                            if response.status_code == 200:
                                noisy_image = cv2.imdecode(np.frombuffer(response.content, np.uint8), cv2.IMREAD_COLOR)
                                # Send Remove Noise Request
                                _, img_encoded = cv2.imencode('.png', noisy_image)
                                denoise_files = {
                                    "file": ("noisy_image.png", io.BytesIO(img_encoded.tobytes()), "image/png")
                                }
                                denoise_response = requests.post(ENDPOINT_REMOVE_NOISE, files=denoise_files)
                                if denoise_response.status_code == 200:
                                    denoised_image = cv2.imdecode(np.frombuffer(denoise_response.content, np.uint8), cv2.IMREAD_COLOR)
                                    # Send Sharpen Image Request
                                    _, denoised_encoded = cv2.imencode('.png', denoised_image)
                                    sharpen_files = {
                                        "file": ("denoised_image.png", io.BytesIO(denoised_encoded.tobytes()), "image/png")
                                    }
                                    sharpen_response = requests.post(ENDPOINT_SHARPEN_IMAGE, files=sharpen_files)
                                    if sharpen_response.status_code == 200:
                                        sharpened_image = cv2.imdecode(np.frombuffer(sharpen_response.content, np.uint8), cv2.IMREAD_COLOR)
                                        st.success("Image processing completed successfully.")
                                    else:
                                        st.error("Error in sharpening the image.")
                                else:
                                    st.error("Error in removing noise from the image.")
                            else:
                                st.error("Error in adding noise to the image.")

            # Display Processed Images if available
            if 'noisy_image' in locals() and 'denoised_image' in locals() and 'sharpened_image' in locals():
                cols = st.columns(3)
                with cols[0]:
                    st.image(noisy_image, caption="Image with Salt & Pepper Noise",
                             channels="BGR", width=image_width)
                with cols[1]:
                    st.image(denoised_image, caption="Image After Noise Removal",
                             channels="BGR", width=image_width)
                with cols[2]:
                    st.image(sharpened_image, caption="Image After Sharpening",
                             channels="BGR", width=image_width)

                folder_name = st.text_input("Enter the name of the saving folder:")
                save_processed = st.button("Save Processed Images")

                if save_processed:
                    if not folder_name:
                        folder_name = "temp_" + uuid.uuid4().hex[:8]
                    else:
                        folder_name = "".join(c for c in folder_name if c.isalnum() or c in (' ', '_', '-')).rstrip()

                    processed_path = os.path.join('processed_dataset', folder_name)
                    os.makedirs(processed_path, exist_ok=True)

                    unique_id = uuid.uuid4().hex
                    cv2.imwrite(os.path.join(processed_path, f'noisy_image_{unique_id}.jpg'), noisy_image)
                    cv2.imwrite(os.path.join(processed_path, f'denoised_image_{unique_id}.jpg'), denoised_image)
                    cv2.imwrite(os.path.join(processed_path, f'sharpened_image_{unique_id}.jpg'), sharpened_image)
                    st.success(f"Images have been processed and saved in the folder processed_dataset/{folder_name}.")

    else:
        st.warning("No images found in the dataset. Please add new faces first.")

# ---------------------- RGB Extraction and Image Operations Section ---------------------- #
elif app_mode == "RGB Extraction & Image Operations":
    st.title("RGB Extraction and Image Operations")

    def extract_rgb(image):
        try:
            # Reset the file pointer to the beginning
            image.seek(0)
            files = {
                "file": ("image.jpg", io.BytesIO(image.read()), "image/jpeg")
            }
            response = requests.post(ENDPOINT_RGB_EXTRACTION, files=files)
            if response.status_code == 200:
                return response.json().get('average_rgb')
            else:
                st.error(response.json().get('detail'))
                return None
        except Exception as e:
            st.error(f"An error occurred during RGB extraction: {e}")
            return None

    def perform_arithmetic_operation(image, operation, value):
        try:
            # Reset the file pointer to the beginning
            image.seek(0)
            files = {
                "file": ("image.jpg", io.BytesIO(image.read()), "image/jpeg")
            }
            data = {
                "operation": operation,
                "value": value
            }
            response = requests.post(ENDPOINT_OPERATION_RGB, files=files, data=data)
            if response.status_code == 200:
                return cv2.imdecode(np.frombuffer(response.content, np.uint8), cv2.IMREAD_COLOR)
            else:
                st.error(response.json().get('detail'))
                return None
        except Exception as e:
            st.error(f"An error occurred during arithmetic operation: {e}")
            return None

    def perform_logic_operation(image1, image2, operation):
        try:
            # Reset the file pointers
            image1.seek(0)
            files = {
                "file1": ("image1.jpg", io.BytesIO(image1.read()), "image/jpeg")
            }
            data = {
                "operation": operation
            }
            if image2:
                image2.seek(0)
                files["file2"] = ("image2.jpg", io.BytesIO(image2.read()), "image/jpeg")
            response = requests.post(ENDPOINT_LOGIC_OPERATION_RGB, files=files, data=data)
            if response.status_code == 200:
                return cv2.imdecode(np.frombuffer(response.content, np.uint8), cv2.IMREAD_COLOR)
            else:
                st.error(response.json().get('detail'))
                return None
        except Exception as e:
            st.error(f"An error occurred during logical operation: {e}")
            return None

    def convert_to_grayscale(image):
        try:
            # Reset the file pointer to the beginning
            image.seek(0)
            files = {
                "file": ("image.jpg", io.BytesIO(image.read()), "image/jpeg")
            }
            response = requests.post(ENDPOINT_GRAYSCALE, files=files)
            if response.status_code == 200:
                return cv2.imdecode(np.frombuffer(response.content, np.uint8), cv2.IMREAD_GRAYSCALE)
            else:
                st.error(response.json().get('detail'))
                return None
        except Exception as e:
            st.error(f"An error occurred during grayscale conversion: {e}")
            return None

    def generate_histograms(image):
        try:
            # Reset the file pointer to the beginning
            image.seek(0)
            files = {
                "file": ("image.jpg", io.BytesIO(image.read()), "image/jpeg")
            }
            response = requests.post(ENDPOINT_HISTOGRAM, files=files)
            if response.status_code == 200:
                histograms = response.json()
                # Construct full URLs for histograms
                grayscale_hist_url = BACKEND_URL + histograms.get('grayscale_histogram')
                color_hist_url = BACKEND_URL + histograms.get('color_histogram')
                return grayscale_hist_url, color_hist_url
            else:
                st.error(response.json().get('detail'))
                return None, None
        except Exception as e:
            st.error(f"An error occurred during histogram generation: {e}")
            return None, None

    def equalize_histogram(image):
        try:
            # Reset the file pointer to the beginning
            image.seek(0)
            files = {
                "file": ("image.jpg", io.BytesIO(image.read()), "image/jpeg")
            }
            response = requests.post(ENDPOINT_EQUALIZE, files=files)
            if response.status_code == 200:
                return cv2.imdecode(np.frombuffer(response.content, np.uint8), cv2.IMREAD_COLOR)
            else:
                st.error(response.json().get('detail'))
                return None
        except Exception as e:
            st.error(f"An error occurred during histogram equalization: {e}")
            return None

    def specify_histogram(image, reference_image):
        try:
            # Reset the file pointers
            image.seek(0)
            reference_image.seek(0)
            files = {
                "file": ("main_image.jpg", io.BytesIO(image.read()), "image/jpeg"),
                "ref_file": ("ref_image.jpg", io.BytesIO(reference_image.read()), "image/jpeg")
            }
            response = requests.post(ENDPOINT_SPECIFY_HISTOGRAM, files=files)
            if response.status_code == 200:
                return cv2.imdecode(np.frombuffer(response.content, np.uint8), cv2.IMREAD_COLOR)
            else:
                st.error(response.json().get('detail'))
                return None
        except Exception as e:
            st.error(f"An error occurred during histogram specification: {e}")
            return None

    def calculate_statistics(image):
        try:
            # Reset the file pointer to the beginning
            image.seek(0)
            files = {
                "file": ("image.jpg", io.BytesIO(image.read()), "image/jpeg")
            }
            response = requests.post(ENDPOINT_STATISTICS, files=files)
            if response.status_code == 200:
                stats = response.json()
                return stats.get('mean_intensity'), stats.get('standard_deviation')
            else:
                st.error(response.json().get('detail'))
                return None, None
        except Exception as e:
            st.error(f"An error occurred during statistics calculation: {e}")
            return None, None

    uploaded_file = st.file_uploader("Upload an image for RGB Extraction and Operations:", type=["png", "jpg", "jpeg"])

    if uploaded_file:
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

        rgb_values = extract_rgb(uploaded_file)
        if rgb_values:
            st.write(f"**Average RGB Values:** R: {rgb_values[0]:.2f}, G: {rgb_values[1]:.2f}, B: {rgb_values[2]:.2f}")

        st.subheader("Arithmetic Operations")
        operation = st.selectbox("Select Operation:", ["None", "Add", "Subtract", "Max", "Min", "Inverse"])
        value = 0
        if operation in ["Add", "Subtract", "Max", "Min"]:
            value = st.number_input("Enter the value for the operation (0-255):", min_value=0, max_value=255, value=0)

        if st.button("Apply Arithmetic Operation"):
            if operation != "None":
                processed_img = perform_arithmetic_operation(uploaded_file, operation.lower(), value)
                print(value)
                if processed_img is not None:
                    st.image(processed_img, caption=f"Image After {operation} Operation", channels="BGR", use_column_width=True)
            else:
                st.warning("No operation selected.")

        st.subheader("Logical Operations")
        logic_operation = st.selectbox("Select Logic Operation:", ["None", "AND", "XOR", "NOT"])
        image2 = None
        if logic_operation in ["AND", "XOR"]:
            image2_file = st.file_uploader("Upload a second image for Logic Operation:", type=["png", "jpg", "jpeg"], key="logic_op_image")
            if image2_file:
                st.image(image2_file, caption="Second Image", use_column_width=True)
                image2 = image2_file

        if st.button("Apply Logic Operation"):
            if logic_operation != "None":
                if logic_operation in ["AND", "XOR"] and not image2:
                    st.error("Please upload a second image for this operation.")
                else:
                    processed_img = perform_logic_operation(uploaded_file, image2, logic_operation.lower())
                    if processed_img is not None:
                        st.image(processed_img, caption=f"Image After {logic_operation} Operation", channels="BGR", use_column_width=True)
            else:
                st.warning("No logic operation selected.")

        st.subheader("Grayscale Conversion")
        if st.button("Convert to Grayscale"):
            gray_img = convert_to_grayscale(uploaded_file)
            if gray_img is not None:
                st.image(gray_img, caption="Grayscale Image", use_column_width=True)

        st.subheader("Histogram Generation")
        if st.button("Generate Histograms"):
            grayscale_hist_url, color_hist_url = generate_histograms(uploaded_file)
            if grayscale_hist_url and color_hist_url:
                # Display histograms using URLs
                st.image(grayscale_hist_url, caption="Grayscale Histogram", use_column_width=True)
                st.image(color_hist_url, caption="Color Histogram", use_column_width=True)

        st.subheader("Histogram Equalization")
        if st.button("Equalize Histogram"):
            equalized_img = equalize_histogram(uploaded_file)
            if equalized_img is not None:
                st.image(equalized_img, caption="Equalized Image", channels="BGR", use_column_width=True)

        st.subheader("Histogram Specification")
        reference_file = st.file_uploader("Upload a reference image for histogram specification:", type=["png", "jpg", "jpeg"], key="hist_spec_ref")
        if st.button("Apply Histogram Specification"):
            if not reference_file:
                st.error("Please upload a reference image for histogram specification.")
            else:
                specified_img = specify_histogram(uploaded_file, reference_file)
                if specified_img is not None:
                    st.image(specified_img, caption="Specified Histogram Image", channels="BGR", use_column_width=True)

        st.subheader("Statistics Calculation")
        if st.button("Calculate Statistics"):
            mean_intensity, std_deviation = calculate_statistics(uploaded_file)
            if mean_intensity is not None and std_deviation is not None:
                st.write(f"**Mean Intensity:** {mean_intensity}")
                st.write(f"**Standard Deviation:** {std_deviation}")
