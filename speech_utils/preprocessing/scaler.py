class Scaler:
    def __init__(self, t_max = 1, t_min = -1):
        self.t_max = t_max
        self.t_min = t_min

    def fit(self, x):
        self.medians = x.median()
        self.stds = x.std()

        self.minimums = self.medians - (3 * self.stds)
        self.maximums = self.medians + (3 * self.stds)

    def transform(self, x):
        return ((x - self.minimums) / (self.maximums - self.minimums)) * (
            self.t_max - self.t_min
        ) + self.t_min