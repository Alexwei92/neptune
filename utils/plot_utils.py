import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2

class DynamicPlot():
    '''
    plot the position in NED frame
    '''
    def __init__(self, fig, axes, max_width=100, label1='line1'):
        self.fig = fig
        self.axes = axes
        self.max_width = max_width
        self.line, = self.axes.plot([], [], label=label1)

    def update(self, x_data, y_data):
        if len(self.line.get_xdata()) <= self.max_width:
            self.line.set_ydata(np.append(self.line.get_ydata(), y_data))
            self.line.set_xdata(np.append(self.line.get_xdata(), x_data))
        else:
            self.line.set_ydata(np.append(np.delete(self.line.get_ydata(), 0), y_data))
            self.line.set_xdata(np.append(np.delete(self.line.get_xdata(), 0), x_data))
        self.axes.relim()
        self.axes.autoscale_view()
        # self.fig.draw()
        plt.pause(1e-5)

# Plot heading in a bar
# yaw_cmd is normalized to [-0.5, 0.5]
def plot_with_heading(image, yaw_cmd):

    height = image.shape[0]
    width = image.shape[1]

    w1 = 0.35
    w2 = 0.65
    h1 = 0.85
    h2 = 0.9
    cv2.rectangle(image, (int(w1*width), int(h1*height)), (int(w2*width), int(h2*height)), (0,0,0), 2)

    bar_width = 10
    center_pos = ((w2-w1)*width - bar_width) * yaw_cmd + 0.5*width
    cv2.rectangle(image, (int(center_pos-bar_width/2),int(0.85*height)), (int(center_pos+bar_width/2),int(0.9*height)), (0,0,255), -1)
    cv2.imshow("mirror", image) 

# plot numpy.ndarray image
def imshow_np(img):
    if torch.is_tensor(img):
        img = img.numpy()
    # input image is in [-1,1] range
    img = ((img + 1.0) / 2.0 * 255.0).astype(np.uint8)
    plt.imshow(img.transpose(2,1,0))
    plt.axis('off')

# plot PIL image
def imshow_PIL(img):
    if torch.is_tensor(img):
        img = img.permute((1,2,0))
    plt.imshow(img)
    plt.axis('off')