#!/usr/bin/env python3

from base_algo import load_mats
from PIL import Image, ImageDraw
import imageio
import numpy as np

def draw_circle(draw, center, radius, fill=None):
    draw.ellipse([(center[0] - radius, center[1] - radius), (center[0] + radius, center[1] + radius)], fill)
    
def draw_electrode(draw, center, fill=None):
    draw_circle(draw, center, 50, fill)

def draw_border(draw):
    border = [(200, 0), (200, 700), (0, 700), (0, 1600), (400, 1600), 
              (800, 1200), (800, 400), (600, 400), (600, 0), (200, 0)]
    for i in range(len(border) - 1):
        draw.line([border[i + 1], border[i]], fill=0, width=5)

def gen_image(freqs):
    im = Image.new('L', (800, 1600), color='white')
    draw = ImageDraw.Draw(im)
    draw_border(draw)

    centers = [(300, 100), (500, 100), (300, 300), (500, 300), (300, 500),
               (500, 500), (700, 500), (600, 600), (300, 700), (500, 700), 
               (700, 700), (200, 800), (400, 800), (600, 800), (100, 900), 
               (300, 900), (500, 900), (700, 900), (200, 1000), (400, 1000),
               (600, 1000), (100, 1100), (300, 1100), (500, 1100), (200, 1200),
               (400, 1200), (600, 1200), (100, 1300), (300, 1300), (500, 1300),
               (100, 1500), (300, 1500)]
    for center, freq in zip(centers, freqs):
        draw_electrode(draw, center, int(freq))

    return np.array(im.resize((400, 800)))

if __name__ == '__main__':
    X_train, _, y_train, _ = load_mats()
    X_train = X_train[1000:1100]
    X_train = np.absolute(X_train)
    X_train = 255 * X_train / np.amax(X_train)

    for i in range(26):
        with imageio.get_writer('./frequency%d.gif' % i, mode='I') as writer:
            for x in X_train:
                writer.append_data(gen_image(x[:,i]))

