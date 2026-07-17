import pygame
import cv2
import numpy as np
import mediapipe as mp
from asl_mlp import ASL_MLP
import torch
import pickle
import pyttsx3
import time

#initilize pyttsx3
tts_engine = pyttsx3.init()

#sets up time and the letter stuff
HOLD_TIME = 0.8
current_letter = ""
hold_start = None
last_spoken_letter = None
letter_accepted = False

# load the mpl which is trained
model = ASL_MLP(in_dim=63, num_classes=24)
model.load_state_dict(torch.load("asl_model.pth"))
model.eval()

# loads the labels so that the letter it predicts go to the right letter
with open("label_encoder.pkl", "rb") as f:
    encoder = pickle.load(f)

#sets up medipipe 
#static image is false because its a live feed
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils  # for drawing the hand skeleton on screen

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

#sets up the camera feed
cap = cv2.VideoCapture(0)

# grab one frame first to get the camera's actual resolution
ret, frame = cap.read()
if not ret:
    raise RuntimeError("Could not read from camera")
frame_height, frame_width = frame.shape[:2]

#initilizes pygame + sets up the screen
pygame.init()
screen = pygame.display.set_mode((frame_width, frame_height))
pygame.display.set_caption("ASL Recognition")
font = pygame.font.SysFont(None, 72)  # fixed: was "fomt"

#sets up clock + running = true
clock = pygame.time.Clock()
running = True

def speak_word(word):
    tts_engine.say(word)
    tts_engine.runAndWait()
    print("Spoken:", word)

word = ""

#while running
while running: 
    #handles stuff to the window
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        elif event.type == pygame.KEYDOWN:

            if event.key == pygame.K_q:
                running = False

            elif event.key == pygame.K_RETURN:
                print("spoke")
                speak_word(word)

            elif event.key == pygame.K_BACKSPACE:
                word = word[:-1]
            
            elif event.key == pygame.K_c:
                word = ""
                print("Cleard")
            elif event.key == pygame.K_SPACE:
                word += " "
                print("added space")

    # gets the frame and checks if it got the frame in ret, and if it didn't get the frame it closes the loop
    ret, frame = cap.read()
    if not ret:
        break

    # converts BGR to RGB for MediaPipe
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # processes the frame into hand keypoints so like at the vital points in the hand
    results = hands.process(rgb)

    letter = ""  # reset every frame so old text doesn't linger when no hand is detected

    if results.multi_hand_landmarks:
        # takes the first detected hand
        hand_landmarks = results.multi_hand_landmarks[0]

        # draws the hand skeleton onto the rgb frame (since that's what gets displayed)
        mp_drawing.draw_landmarks(rgb, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # converts points into an array of (x, y, z) - 21 points
        pts = np.array(
            [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark],
            dtype=np.float32
        )

        # makes the wrist the zero point so that it is easier to do the math
        wrist = pts[0]
        pts = pts - wrist

        # takes the length from the wrist to the base of the middle finger, then
        # makes a scale of your hand so that no matter how close or far you are from the camera,
        # it will be accurate with the lengths and can size the lengths up
        scale = np.linalg.norm(pts[9])

        if scale > 0:

            pts = pts / scale
            flat = pts.flatten()  # 63 values (x, y, z for 21 points)

            # converts the array into torch then sees the output
            input_data = torch.tensor(flat, dtype=torch.float32).unsqueeze(0)

            with torch.no_grad():
                output = model(input_data)

            prediction = torch.argmax(output, dim=1).item()

            letter = encoder.inverse_transform([prediction])[0]

            print("ASL Letter:", letter)


            if letter == current_letter:

                if hold_start is None:
                    hold_start = time.time()

                elif time.time() - hold_start >= HOLD_TIME and not letter_accepted:
                    word += letter
                    letter_accepted = True

            else:
                current_letter = letter
                hold_start = time.time()
                letter_accepted = False

    else:
        current_letter = ""
        hold_start = None
        letter_accepted = False


    # pygame expects (width, height, channels) with axes swapped vs numpy's (height, width, channels)
    frame_surface = pygame.surfarray.make_surface(np.transpose(rgb, (1, 0, 2)))

    screen.blit(frame_surface, (0, 0))


    if letter:
        text_surface = font.render(letter, True, (0, 0, 0))  
        screen.blit(text_surface, (20, 20))


    word_surface = font.render(word, True, (255, 0, 0))
    screen.blit(word_surface, (20, 100))


    pygame.display.flip()
    clock.tick(30)  # caps at ~30fps so it doesn't run wildly faster than the camera

cap.release()
pygame.quit()