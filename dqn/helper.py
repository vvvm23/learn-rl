class EpisilonAnnealer:
    def __init__(self, start_eps: float, end_eps: float, end_frame: int):
        self.start_eps = start_eps
        self.end_eps = end_eps
        self.end_frame = end_frame

    def get(self, step):
        return max(self.end_eps, self.start_eps - step / self.end_frame)
