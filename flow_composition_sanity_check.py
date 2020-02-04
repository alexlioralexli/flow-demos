# sanity check to self about composing layers of single gaussians
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

if __name__ == '__main__':
    plt.figure(figsize=(15, 4))
    x = np.linspace(-3, 3, 1000)
    y = norm.cdf(x)

    locs = [0.1, 0.3, 0.5, 0.7, 0.9]
    scales = [0.1, 0.4, 0.7]
    for i, scale in enumerate(scales):
        plt.subplot(1,len(scales),i+1)
        plt.plot(x, norm.pdf(x), label='N(0,1)')
        for loc in locs:
            plt.plot(x, norm.pdf(x) * norm.pdf(y, loc=loc, scale=scale), label=f'loc={loc}')
        plt.legend()
        plt.title(f'scale={scale}')