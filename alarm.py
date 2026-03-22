import threading

try:
    import winsound
except ModuleNotFoundError:
    winsound = None


class AlarmPlayer:
    def __init__(self, sound_file_path: str):
        self.sound_file_path = sound_file_path
        self.is_playing = False
        self.lock = threading.Lock()

    @property
    def audio_mode(self) -> str:
        return "Local system alarm" if winsound is not None else "Silent fallback"

    def update(self, should_play: bool):
        if winsound is None:
            self.is_playing = False
            return

        with self.lock:
            if should_play and not self.is_playing:
                winsound.PlaySound(
                    self.sound_file_path,
                    winsound.SND_ASYNC | winsound.SND_FILENAME | winsound.SND_LOOP,
                )
                self.is_playing = True
            elif not should_play and self.is_playing:
                winsound.PlaySound(None, winsound.SND_PURGE)
                self.is_playing = False

    def stop(self):
        self.update(False)
