import os
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
import pygame
import asyncio

class AsyncAudioDriver:
    def __init__(self, AudioFeedbackPublisher, Verbose=False, sample_rate=44100, buffer=1024, max_channels=16):
        pygame.mixer.pre_init(sample_rate, -16, 2, buffer)
        pygame.mixer.init()
        pygame.mixer.set_num_channels(max_channels)
        self.AudioFeedbackPublisher = AudioFeedbackPublisher

        base = "/home/turkeysquad/ME450/MainDesign/Canine/AudioFiles"

        self.Verbose = Verbose
        self._current = None
        self._playing = False

        self.sounds = {
            "BarkShort":     pygame.mixer.Sound(f"{base}/BarkShort.mp3"),
            "BarkBurst":     pygame.mixer.Sound(f"{base}/BarkBurst.mp3"),
            "Growl":         pygame.mixer.Sound(f"{base}/Growl.mp3"),
            "AggressiveBark":pygame.mixer.Sound(f"{base}/AggressiveBark.mp3"),
        }

    # ------------------------------------------------------------------
    # Internal — polls pygame to detect natural playback completion
    # ------------------------------------------------------------------

    async def _monitor(self):
        while True:
            await asyncio.sleep(0.2)
            if self._playing and not pygame.mixer.get_busy():
                await self.AudioFeedbackPublisher.publish({"Id": "System", "Type": "Audio", "Data": "Stop"})
                self._current = None
                self._playing = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def play(self, name, volume=1.0):
        if name == "Stop":
            pygame.mixer.stop()
            self._current = None
            self._playing = False
            if self.Verbose: print("AudioDriver: Stop audio received")
            return
        sound = self.sounds.get(name)
        if sound:
            pygame.mixer.stop()
            sound.set_volume(volume)
            sound.play()
            self._current = name
            self._playing = True
            if self.Verbose: print("AudioDriver: Playing", name)

    def stop_all(self):
        pygame.mixer.stop()
        self._current = None
        self._playing = False
        if self.Verbose: print("AudioDriver: Stopped all audio")

    def shutdown(self):
        pygame.mixer.stop()
        pygame.mixer.quit()