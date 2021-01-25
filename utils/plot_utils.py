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
    # input image is normalized to [-1,1]
    img = ((img + 1.0) / 2.0 * 255.0).astype(np.uint8)
    plt.imshow(cv2.cvtColor(img.transpose(2,1,0), cv2.COLOR_BGR2RGB))
    plt.axis('off')

# plot PIL image
def imshow_PIL(img):
    if torch.is_tensor(img):
        img = img.permute((1,2,0))
    plt.imshow(img)
    plt.axis('off')

# plot generated and original figure
def plot_generate_figure(output, original, disp_N=6):
    fig = plt.figure()
    for i in range(disp_N):
        plt.subplot(2, disp_N, i+1)
        imshow_np(original[i,:])
        plt.subplot(2, disp_N, i+1+disp_N)
        imshow_np(output[i,:])

# plot training losses history
def plot_train_losses(train_history):
    train_counter, train_losses, train_MSE_losses, train_KLD_losses = train_history

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)
    ax1.plot(train_counter, train_KLD_losses, color='blue')
    ax1.legend(['KLD Loss'], loc='upper right')
    ax2.plot(train_counter, train_MSE_losses, color='blue')
    ax2.legend(['MSE Loss'], loc='upper right')
    ax2.set_ylabel('Loss')
    ax3.plot(train_counter, train_losses, color='blue')
    ax3.legend(['Total Loss'], loc='upper right')
    plt.xlabel('# of training samples')
    plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))

# plot training losses history
def plot_train_losses2(train_history):
    train_counter, train_losses = train_history

    fig, ax = plt.subplots(1,1)
    ax.plot(train_counter, train_losses, color='blue')
    ax.legend(['Total Loss'], loc='upper right')
    plt.xlabel('# of training samples')
    plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))