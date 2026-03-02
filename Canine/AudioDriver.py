import os
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
import asyncio
import pygame

class AsyncAudioDriver:
    def __init__(self, sample_rate=44100, buffer=1024, max_channels=16):
        pygame.mixer.pre_init(sample_rate, -16, 2, buffer)
        pygame.mixer.init()
        pygame.mixer.set_num_channels(max_channels)

        base = "/home/meathead/ME450/MainDesign/Canine/AudioFiles"
        self.sounds = {
            "startup": pygame.mixer.Sound(f"{base}/TEMP_Bark1.wav"),
            "alert":   pygame.mixer.Sound(f"{base}/TEMP_Bark2.wav"),
            "howl":    pygame.mixer.Sound(f"{base}/TEMP_Bark3.wav"),
        }

    def play(self, name, volume=1.0):
        sound = self.sounds.get(name)
        if sound:
            sound.set_volume(volume)
            sound.play()

    def stop_all(self):
        pygame.mixer.stop()

    def shutdown(self):
        pygame.mixer.stop()
        pygame.mixer.quit()