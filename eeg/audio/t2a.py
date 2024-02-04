import tkinter as tk
import threading
from pylsl import StreamInlet, resolve_stream
import csv
import random
import time
from gtts import gTTS
import pygame
import io

# Global variables
current_label = 1
sample_data = []

def save_to_csv(sample, label):
    with open('audio.csv', 'a', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        row_data = [label] + list(sample)
        csv_writer.writerow(row_data)

def speak_sentence(sentence):
    tts = gTTS(text=sentence, lang='en')
    fp = io.BytesIO()
    tts.write_to_fp(fp)
    fp.seek(0)
    pygame.mixer.init()
    pygame.mixer.music.load(fp)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        time.sleep(1)

def collect_eeg_data():
    streams = resolve_stream('type', 'EEG')
    inlet = StreamInlet(streams[0])

    while True:
        sample, _ = inlet.pull_sample()
        sample_data.append(sample)

def main():
    window = tk.Tk()
    window.title('EEG Data Annotation')
    width = window.winfo_screenwidth()
    height = window.winfo_screenheight()
    window.geometry(f"{width}x{height}")
    window.configure(bg='white')

    sentences = ["Hello, how are you?", "This is a test.", "Reading sentences for EEG data."]  # Example sentences
    sentence_index = 0

    def read_next_sentence():
        nonlocal sentence_index
        if sentence_index < len(sentences):
            speak_thread = threading.Thread(target=speak_sentence, args=(sentences[sentence_index],))
            speak_thread.daemon = True
            speak_thread.start()
            sentence_index += 1

    start_button = tk.Button(window, text="Read Next Sentence", command=read_next_sentence)
    start_button.grid(row=7, column=3, pady=20)

    eeg_thread = threading.Thread(target=collect_eeg_data)
    eeg_thread.daemon = True
    eeg_thread.start()

    window.mainloop()

if __name__ == '__main__':
    main()
