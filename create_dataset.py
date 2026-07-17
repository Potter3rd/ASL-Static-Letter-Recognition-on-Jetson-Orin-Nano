import cv2
import numpy as np
import os
import csv
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,      # treat each image independently (no video tracking)
    max_num_hands=1,
    min_detection_confidence=0.3  # a bit lenient since dataset hands vary a lot
)

dataset_path = os.path.expanduser("~/asl_alphabet_data/asl_alphabet_train/asl_alphabet_train")


# once working, expand to the full set (skips J and Z, which involve motion):
letters = ["A","B","C","D","E","F","G","H","I","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y"]

rows = []

total_images = 0
saved_images = 0


for letter in letters:

    folder = os.path.join(dataset_path, letter)

    if not os.path.isdir(folder):
        print("Folder not found:", folder)
        continue

    print("\nProcessing letter:", letter)

    for file in os.listdir(folder):

        if not file.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        total_images += 1

        image_path = os.path.join(folder, file)
        frame = cv2.imread(image_path)

        if frame is None:
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        if not results.multi_hand_landmarks:
            print(file, "no hand found")
            continue

        hand_landmarks = results.multi_hand_landmarks[0]

        pts = np.array(
            [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark],
            dtype=np.float32
        )  # shape: (21, 3)

        print(file, "found", len(pts), "landmarks")

        # Wrist as origin
        pts = pts - pts[0]

        # Normalize size using wrist -> middle-MCP (landmark 9) distance
        scale = np.linalg.norm(pts[9])

        if scale > 0:
            pts = pts / scale
            flat = pts.flatten()  # 63 values (x,y,z for 21 points)

            rows.append(list(flat) + [letter])
            saved_images += 1


print("\nFinished")
print("Total images checked:", total_images)
print("Saved examples:", saved_images)

with open("asl_dataset_mediapipe.csv", "w", newline="") as f:
    writer = csv.writer(f)
    header = [f"x{i}" for i in range(63)] + ["label"]
    writer.writerow(header)
    writer.writerows(rows)

print("CSV created!")