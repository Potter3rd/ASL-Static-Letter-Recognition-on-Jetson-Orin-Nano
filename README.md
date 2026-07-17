 ## ASL Static Letter Recognition on Jetson Orin Nano

This is a project I made that reads American Sign Language letters in real time using a Jetson Orin Nano. It uses MediaPipe to track your hand, runs the points through a model I trained, and shows the letter it thinks you're signing. You can hold a sign for a bit to add that letter to a word, and when you're done it'll read the word out loud.

# What it does

- Tracks your hand in real time using MediaPipe
- Uses a model I trained that gets 99.27% accuracy on 24 letters
- If you hold a sign for about 0.8 seconds it adds that letter to the word
- Shows the camera feed with the hand points drawn on it plus the letter it's reading
- You can backspace or clear the word if you mess up
- Hit enter and it speaks the word out loud, no internet needed since its all offline

# What you need

- Jetson Orin Nano
- a USB webcam
- a monitor + keyboard/mouse (or just SSH/remote into it)
- speakers or bluetooth headphones/speaker so it can actually talk

# Stuff to install

```bash
pip3 install torch opencv-python mediapipe pygame pyttsx3
sudo apt install espeak -y
```

# How to run it

```bash
python3 AslReader.py
```

**Controls:**

- hold a sign (~0.8s) → adds the letter to the word
- `Enter` → says the word out loud
- `Backspace` → deletes the last letter
- `C` → clears the whole word
- `Q` → quits

# How it actually works

1. First it grabs the frame using openCV
2. Then MediaPipe finds 21 points on your hand (x, y, z) for each point
3. Then I zero everything out based on the wrist point and scale it by hand size, so it doesn't matter how close/far you are from the camera or where your hand is in frame
4. The I take those 63 numbers (21 points times x/y/z) feed it into the MLP model, which spits out a letter it thinks it is.
5. Then it only actually adds the letter once you've held the same sign for like 0.8 seconds, so it doesn't just spam the same letter over and over
6. Finaly when you hit enter it sends the built up word to pyttsx3 and it reads it out loud, fully offline

# Files

- `AslReader.py` — the main code, runs the camera + UI + speech
- `train_mlp.py` — trains the MLP model
- `create_dataset.py` — turns the training images into hand keypoints
- `asl_mlp.py` — the model class itself
- `asl_model.pth` — the trained weights
- `label_encoder.pkl` — maps the model's output back to actual letters

## About the model

- It is trained on the Kaggle ASL Alphabet dataset (~87k images, 29 classes) (URL for dataset is in the citations)
- takes in 63 numbers per frame (21 hand points × x/y/z)
- only does 24 letters — J and Z are left out since those need motion, not just a still pose
- gets 99.27% accuracy on the test set

## Limitations

- can't do J or Z since they're motion-based signs
- only works with one hand at a time
- accuracy can depend on lighting and how your hand is framed in the camera, since its just reading a single pose not actually translating full sign language

## Credits

- hand tracking from [MediaPipe](https://developers.google.com/mediapipe)
- training data from the [Kaggle ASL Alphabet dataset](https://www.kaggle.com/datasets/grassknoted/asl-alphabet)
- @misc{https://www.kaggle.com/grassknoted/aslalphabet_akash nagaraj_2018,
  title={ASL Alphabet},
  url={https://www.kaggle.com/dsv/29550},
  DOI={10.34740/KAGGLE/DSV/29550},

  ## Demo Video for la Nvidia Peeps
  [Demo](https://docs.google.com/videos/d/15DZFo7KidsJ5vDPuXyyoFFq0_m5EX7zb0PpcY3Il510/edit?usp=drive_link)
