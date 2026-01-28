import os
import matplotlib.pyplot as plt

class PlotIO:
    """Policy for saving + optionally showing plots."""
    def __init__(self, output_dir: str, show: bool = True, dpi: int = 150):
        self.output_dir = output_dir
        self.show = show
        self.dpi = dpi
        os.makedirs(self.output_dir, exist_ok=True)

    def save(self, filename: str) -> str:
        path = os.path.join(self.output_dir, filename)
        plt.tight_layout()
        plt.savefig(path, dpi=self.dpi)
        if self.show:
            plt.show()
        plt.close()
        return path
