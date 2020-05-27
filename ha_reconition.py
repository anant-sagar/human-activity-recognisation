import numpy as np
import argparse
import imutils
import sys
import cv2
import streamlit as st


def execute(input=0):
    CLASSES = open('labels/action_recognition_kinetics.txt').read().strip().split("\n")
    SAMPLE_DURATION = 16
    SAMPLE_SIZE = 112
    INPUT = input
    with st.spinner('loading AI model in memory'):
        net = cv2.dnn.readNet('model/resnet-34_kinetics.onnx')
        st.success("model loading in memory")
    vs = cv2.VideoCapture(INPUT)

    while True:
        frames = []
        for i in range(0, SAMPLE_DURATION):
            (grabbed, frame) = vs.read()

            if not grabbed:
                st.error("No input was found")
                sys.exit(0)

            frame = imutils.resize(frame,width=400)
            frames.append(frame)
        blob = cv2.dnn.blobFromImages(
                                        frames,
                                        1.0,
                                        (SAMPLE_SIZE,SAMPLE_SIZE),
                                        (114.7748, 107.7354, 99.4750),
                                        swapRB=True,
                                        crop=True
                                    )
        blob = np.transpose(blob, (1, 0, 2, 3))
        blob = np.expand_dims(blob, axis = 0)
        net.setInput(blob)
        outputs = net.forward()
        label = CLASSES[np.argmax(outputs)]
        
        for frame in frames:
            cv2.rectangle(frame, (0, 0), (300, 40), (0, 0, 0), -1)
            cv2.putText(frame, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,0.8, (255, 255, 255), 2)
        cv2.imshow("Activity Recognition", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    vs.release()
    cv2.destroyAllWindows()
    return True
