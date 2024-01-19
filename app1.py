import streamlit as st
import cv2
import numpy as np
import io
st.title("CARPET MATCHING APP")
# Upload the images
uploaded_file1 = st.file_uploader("Choose Master Image ", type=["jpg", "jpeg", "png"])
uploaded_file2 = st.file_uploader("Choose Production Image", type=["jpg", "jpeg", "png"])

if uploaded_file1 is not None and uploaded_file2 is not None:
    file_bytes1 = np.asarray(bytearray(uploaded_file1.read()), dtype=np.uint8)
    file_bytes2 = np.asarray(bytearray(uploaded_file2.read()), dtype=np.uint8)

    img1 = cv2.imdecode(file_bytes1, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imdecode(file_bytes2, cv2.IMREAD_GRAYSCALE)

    if st.button('Run'):
        # Initiate SIFT detector
        sift = cv2.SIFT_create()

        # find the keypoints and descriptors with SIFT
        kp1, des1 = sift.detectAndCompute(img1,None)
        kp2, des2 = sift.detectAndCompute(img2,None)

        # BFMatcher with default params
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1,des2, k=2)

        # Apply ratio test
        good = []
        non_matching = []
        for m,n in matches:
            if m.distance < 0.75*n.distance:
                good.append([m])
            else:
                non_matching.append([m])

        # Draw the good matches
        img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        # Draw the non-matching features
        img4 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,non_matching,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        # Calculate the percentage of match and non-match
        percentage_match = len(good) / len(matches) * 100
        percentage_non_match = len(non_matching) / len(matches) * 100

        st.write(f'Percentage match: {percentage_match:.2f}%')
        st.write(f'Percentage non-match: {percentage_non_match:.2f}%')

        st.image(img3, caption='Good Matches', use_column_width=True)
        st.image(img4, caption='Bad Matches ', use_column_width=True)
