import os
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
import pygame

class AsyncAudioDriver:
    def __init__(self, Verbose=False, sample_rate=44100, buffer=1024, max_channels=16):
        pygame.mixer.pre_init(sample_rate, -16, 2, buffer)
        pygame.mixer.init()
        pygame.mixer.set_num_channels(max_channels)

        base = "/home/turkeysquad/ME450/MainDesign/Canine/AudioFiles"
        
        self.Verbose = Verbose

        self.sounds = {
            "BarkShort": pygame.mixer.Sound(f"{base}/BarkShort.mp3"),
            "BarkBurst":   pygame.mixer.Sound(f"{base}/BarkBurst.mp3"),
            "Growl":    pygame.mixer.Sound(f"{base}/Growl.mp3"),
            "AggressiveBark":    pygame.mixer.Sound(f"{base}/AggressiveBark.mp3"),
        }
    
    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    
    def play(self, name, volume=1.0):    
        sound = self.sounds.get(name)
        if sound:
            sound.set_volume(volume)
            sound.play()
            if(self.Verbose): print("AudioDriver: Playing ", name)

    def stop_all(self):
        pygame.mixer.stop()

    def shutdown(self):
        pygame.mixer.stop()
        pygame.mixer.quit()
