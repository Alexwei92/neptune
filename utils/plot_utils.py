import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
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

    def reset(self):
        self.line.set_xdata(np.array([]))
        self.line.set_ydata(np.array([]))

# Plot image with cmd
# input is normalized to [-0.5, 0.5]
def plot_with_cmd(win_name, image, input, is_expert):
    height = image.shape[0]
    width = image.shape[1]
    # plot box
    w1 = 0.35 # [0,1]
    w2 = 0.65 # [0,1]
    h1 = 0.85 # [0,1]
    h2 = 0.9  # [0,1]
    cv2.rectangle(image, (int(w1*width), int(h1*height)), (int(w2*width), int(h2*height)), (0,0,0), 2) # black
    # plot bar
    bar_width = 5 # pixel
    center_pos = ((w2-w1)*width - bar_width) * input + 0.5*width
    if is_expert:
        color = (0,0,255) # red
    else:
        color = (255,0,0) # blue
    cv2.rectangle(image, (int(center_pos-bar_width/2),int(h1*height)), (int(center_pos+bar_width/2),int(h2*height)), color, -1)
    # plot center line
    cv2.line(image, (int(0.5*width),int(h1*height)), (int(0.5*width),int(h2*height)), (255,255,255), 1) # white
    cv2.imshow(win_name, image) 

# Plot image with cmds to compare
# input is normalized to [-0.5, 0.5]
def plot_with_cmd_compare(win_name, image, pilot_input, agent_input):
    height = image.shape[0]
    width = image.shape[1]
    # plot box
    w1 = 0.35 # [0,1]
    w2 = 0.65 # [0,1]
    h1 = 0.85 # [0,1]
    h2 = 0.9  # [0,1]
    cv2.rectangle(image, (int(w1*width), int(h1*height)), (int(w2*width), int(h2*height)), (0,0,0), 2) # black
    # plot bar
    bar_width = 5 # pixel
    pilot_center_pos = ((w2-w1)*width - bar_width) * pilot_input + 0.5*width
    agent_center_pos = ((w2-w1)*width - bar_width) * agent_input + 0.5*width
    # Pilot input
    cv2.rectangle(image, (int(pilot_center_pos-bar_width/2),int(h1*height)), (int(pilot_center_pos+bar_width/2),int(h2*height)), (0,0,255), -1)
    # Agent input
    cv2.rectangle(image, (int(agent_center_pos-bar_width/2),int(h1*height)), (int(agent_center_pos+bar_width/2),int(h2*height)), (255,0,0), -1)
    # plot center line
    cv2.line(image, (int(0.5*width),int(h1*height)), (int(0.5*width),int(h2*height)), (255,255,255), 1) # white
    cv2.imshow(win_name, image) 

# Plot image with heading
def plot_with_heading(win_name, image, input, is_expert):
    height = image.shape[0]
    width = image.shape[1]
    # plot box
    w1 = 0.35 # [0,1]
    w2 = 0.65 # [0,1]
    h1 = 0.85 # [0,1]
    h2 = 0.9  # [0,1]
    FOV = 70 # deg
    
    # plot heading
    center_pos = (input/FOV + 1) * (0.5*width)
    if is_expert:
        color = (0,0,255) # red
    else:
        color = (255,0,0) # blue
    cv2.line(image, (int(center_pos),int(0)), (int(center_pos),int(height)), color, 2)
    # plot center line
    cv2.line(image, (int(0.5*width),int(0)), (int(0.5*width),int(height)), (255,255,255), 1) # white
    cv2.imshow(win_name, image) 

# Plot image without heading
def plot_without_heading(win_name, image):
    cv2.imshow(win_name, image)

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
def plot_generate_figure(output, original, N):
    fig = plt.figure()
    for i in range(N):
        plt.subplot(2, N, i+1)
        imshow_np(original[i,:])
        plt.subplot(2, N, i+1+N)
        imshow_np(output[i,:])

# plot training losses history
def plot_train_losses(train_history):
    if len(train_history) == 4:
        # VAE result
        epoch, total_losses, MSE_losses, KLD_losses = train_history

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)
        ax1.plot(epoch, KLD_losses, color='blue')
        ax1.legend(['KLD Loss'], loc='upper right')
        ax2.plot(epoch, MSE_losses, color='blue')
        ax2.legend(['MSE Loss'], loc='upper right')
        ax2.set_ylabel('Loss')
        ax3.plot(epoch, total_losses, color='blue')
        ax3.legend(['Total Loss'], loc='upper right')
        ax3.xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.xlabel('Epoch')
        # plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
        
    elif len(train_history) == 2:
        # latent result
        epoch, total_losses = train_history

        fig, ax = plt.subplots(1,1)
        ax.plot(epoch, total_losses, color='blue')
        ax.legend(['Loss'], loc='upper right')
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.xlabel('Epoch')
        # plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
        plt.title('NN Control training result')
    else:
        pass