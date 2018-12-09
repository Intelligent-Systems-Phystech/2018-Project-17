#!/usr/bin/env python3

from base_algo import load_mats
from main_algo import centers
from PIL import Image, ImageDraw
import imageio
import numpy as np

gif_centers = centers * 100

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

    for center, freq in zip(gif_centers, freqs):
        draw_electrode(draw, center, int(freq))

    return np.array(im.resize((400, 800)))

if __name__ == '__main__':
    X_train, _, y_train, _ = load_mats()
    X_train = X_train[1000:1100]
    X_train = np.absolute(X_train)
    X_train = 255 * X_train / np.amax(X_train)

    for i in range(26):
        with imageio.get_writer('../report/frequency%d.gif' % i, mode='I') as writer:
            for x in X_train:
                writer.append_data(gen_image(x[:,i]))

