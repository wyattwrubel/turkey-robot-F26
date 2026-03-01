import os
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"

import asyncio
import pygame

class AsyncAudioDriver:
    def __init__(self, sample_rate=44100, buffer=512, max_channels=16):
        pygame.mixer.pre_init(sample_rate, -16, 2, buffer)
        pygame.init()
        pygame.mixer.set_num_channels(max_channels)

        self._sounds = {}
        self._queue = asyncio.Queue()
        self._running = False
        self._worker_task = None

    def register(self, name, filepath):
        self._sounds[name] = pygame.mixer.Sound(filepath)

    async def play(self, name, volume=1.0):
        await self._queue.put((name, volume))

    async def start(self):
        if not self._running:
            self._running = True
            self._worker_task = asyncio.create_task(self._worker())

    async def stop(self):
        self._running = False
        if self._worker_task:
            await self._worker_task
        pygame.mixer.stop()

    async def shutdown(self):
        await self.stop()
        pygame.mixer.quit()

    async def _worker(self):
        while self._running:
            try:
                name, volume = await asyncio.wait_for(self._queue.get(), timeout=0.1)
            except asyncio.TimeoutError:
                continue

            sound = self._sounds.get(name)
            if sound:
                sound.set_volume(volume)
                sound.play()

async def main():
    audio = AsyncAudioDriver()

    audio.register("startup", "/home/meathead/ME450/MainDesign/Canine/AudioFiles/TEMP_Bark1.mp3")
    audio.register("alert", "/home/meathead/ME450/MainDesign/Canine/AudioFiles/TEMP_Bark2.mp3")
    audio.register("howl", "/home/meathead/ME450/MainDesign/Canine/AudioFiles/TEMP_Bark3.mp3")

    await audio.start()

    await audio.play("startup")
    await asyncio.sleep(0.2)
    await audio.play("alert")
    await audio.play("howl")

    await asyncio.sleep(15)
    await audio.shutdown()

if __name__ == "__main__":
    asyncio.run(main())