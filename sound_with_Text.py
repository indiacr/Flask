import pickle
import cv2
import mediapipe as mp
import numpy as np
import pygame
import tempfile

class HandGestureRecognition:
    def __init__(self):
        self.model_dict = pickle.load(open('./model.p', 'rb'))
        self.model = self.model_dict['model']
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.hands = self.mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)
        self.labels_dict = {0: 'L', 1: 'A', 2: 'B', 3: 'C', 4: 'V', 5: 'W', 6: 'Y'}
        pygame.mixer.init()
        self.previous_prediction = None

    def process_frame(self, frame):
        data_aux = []
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style())

                for landmark in hand_landmarks.landmark:
                    x = landmark.x
                    y = landmark.y
                    data_aux.extend([x, y])

        if data_aux:
            prediction = self.model.predict([np.asarray(data_aux)])
            predicted_value = int(prediction[0])
            if predicted_value in self.labels_dict:
                predicted_character = self.labels_dict[predicted_value]
                if predicted_character != self.previous_prediction:
                    self.previous_prediction = predicted_character
                    return predicted_character
        return None
