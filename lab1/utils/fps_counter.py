import time

class FPS:
    def __init__(self) -> None:
        self.prev_time = 0
        self.curr_time = 0

    
    def tick(self):
        self.curr_time = time.time()


    def get_fps(self):
        fps = 1 / (self.curr_time - self.prev_time)
        self.prev_time = self.curr_time
        return str(int(fps))