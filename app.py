import streamlit as st
import cv2
from PIL import Image, ImageEnhance
import numpy as np 
import os 

face_cascade = cv2.CascadeClassifier(r'D:/New folder/python project/New folder/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(r'D:/New folder/python project/New folder/haarcascade_eye.xml')

def detect_faces(our_image):
    new_img = np.array(our_image.convert("RGB"))
    faces = face_cascade.detectMultiScale(new_img, 1.1 , 2)
    for(x, y ,w ,h) in faces:
        cv2.rectangle(new_img,(x,y), (x+w, y+h), (255, 0, 0), 2)
    return new_img, faces
    
def detect_eyes(our_image):
    new_img = np.array(our_image.convert("RGB"))
    eyes = eye_cascade.detectMultiScale(new_img, 1.3 , 5)
    for(x, y ,w ,h) in eyes:
        cv2.rectangle(new_img,(x,y), (x+w, y+h), (0, 255, 0), 2)
    return new_img, eyes
    
def cartoonize_image(our_image):
    new_img = np.array(our_image.convert("RGB"))
    gray = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 9, 9 )
    color = cv2.bilateralFilter(new_img, 9 ,300 ,300)
    cartoon = cv2.bitwise_and(color, color, mask = edges)
    return cartoon
    
def cannize_image(our_image):
    new_img = np.array(our_image.convert("RGB"))
    img = cv2.GaussianBlur(new_img, (13, 13), 0)
    canny = cv2.Canny(img, 100 ,150)
    return canny
    

def main():
    st.title('IMAGE EDITING APP' )
    st.text('Edit your Image in a Fast and Simple way')
    
    activities = ['Detection','About']
    choice = st.sidebar.selectbox('Select activity', activities)
    
    if choice == 'Detection':
        st.subheader('face Detection')
        image_file = st.file_uploader('upload Image', type = ['jpg','jpeg','png'])
 
        if image_file is not None:
            our_image = Image.open(image_file)
            st.text('Original Image')
            st.image(our_image)
            
            enhance_type = st.sidebar.radio("Enhance type",['Original','Grey-scale','Contrast','Brightness','Blurring',])
            
            if enhance_type == 'Grey-scale':
                img = np.array(our_image.convert('RGB'))
                gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
                st.image(gray)
                
            elif enhance_type == 'Contrast':
                rate = st.sidebar.slider("Contrast", 0.5, 6.0)
                enhancer = ImageEnhance.Contrast(our_image)
                enhanced_img = enhancer.enhance(rate)
                st.image(enhanced_img)
                
            elif enhance_type == 'Brightness':
                rate = st.sidebar.slider("Brightness", 0.5, 8.0)
                enhancer = ImageEnhance.Brightness(our_image)
                enhanced_img = enhancer.enhance(rate)
                st.image(enhanced_img)
                
            elif enhance_type == 'Blurring':
                rate = st.sidebar.slider("Blurring", 0.5, 7.0)
                blurred_img = cv2.GaussianBlur(np.array(our_image),(15,15),rate)
                st.image(blurred_img)
                
            elif enhance_type == 'Sharpness':
                rate = st.sidebar.slider("Sharpness", 0.0, 14.0)
                enhancer = ImageEnhance.Sharpness(our_image)
                enhanced_img = enhancer.enhance(rate)
                st.image(enhanced_img)
                
            elif enhance_type == 'Original':
                st.image(our_image, width = 300)
                
            else :
                st.image(our_image,width = 300)
                
        tasks = ["Faces","Eyes","Cartoonize","Cannize"]
        features_choice = st.sidebar.selectbox("Find Features", tasks)
        if st.button("Process"):
            if features_choice == "Faces":
                result_img, result_face = detect_faces(our_image)
                st.image(result_img)
                st.success("Found {} faces".format(len(result_face)))
                
            elif features_choice == "Eyes":
                result_img, result_eyes = detect_eyes(our_image)
                st.image(result_img)
                st.success("Found {} eyes".format(len(result_eyes)))
                
            elif features_choice == "Cartoonize":
                result_img = cartoonize_image(our_image)
                st.image(result_img)
                
            elif features_choice == "Cannize":
                result_img = cannize_image(our_image)
                st.image(result_img)
                            
    elif choice == 'About':
        st.subheader('About the developer')
        st.markdown('Built by [ARUN SARA](https://www.linkedin.com/in/arun-sara-3916531aa/)')
        st.text('Hey! this project is made by Arun Sara,Ankit Nagar,Ashwin Jaiswal.')
            
            
if __name__=='__main__':
    main()
