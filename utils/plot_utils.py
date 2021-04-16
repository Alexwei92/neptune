import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import torch
import cv2
import torchvision.utils as vutils

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
def plot_with_cmd(win_name, image_raw, input, is_expert):
    image = image_raw.copy()
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

# Plot image with cmd and return
# input is normalized to [-0.5, 0.5]
def plot_with_cmd_replay(win_name, image_raw, input, is_expert):
    image = image_raw.copy()
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

    return image

# Plot image with cmds to compare
# input is normalized to [-0.5, 0.5]
def plot_with_cmd_compare(win_name, image_raw, pilot_input, agent_input):
    image = image_raw.copy()
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
def plot_with_heading(win_name, image_raw, input, is_expert):
    image = image_raw.copy()
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
    # input image is normalized to [-1.0,1.0]
    img = ((img + 1.0) / 2.0 * 255.0).astype(np.uint8)
    # input image is normalized to [0.0,1.0]
    # img = (img * 255.0).astype(np.uint8)
    plt.imshow(cv2.cvtColor(img.transpose(2,1,0), cv2.COLOR_BGR2RGB))
    plt.axis('off')

# plot PIL image
def imshow_PIL(img):
    if torch.is_tensor(img):
        img = img.permute((1,2,0))
    plt.imshow(img)
    plt.axis('off')

# plot generated and original figure
def plot_generate_figure(sample_img, generated_img):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    grid_img1 = vutils.make_grid(sample_img, normalize=True, range=(-1,1))
    ax1.imshow(grid_img1.permute(1,2,0))
    ax1.set_axis_off()
    grid_img2 = vutils.make_grid(generated_img, normalize=True, range=(-1,1))
    ax2.imshow(grid_img2.permute(1,2,0))
    ax2.set_axis_off()
    plt.tight_layout()

# plot training losses history
def plot_train_losses(train_history, save_path=None):
    iteration, loss_history_raw = train_history
    loss_history = loss_history_raw.copy()
    if 'kld_loss_z' in loss_history:
        N = len(loss_history) - 1
        del loss_history['kld_loss_z']
    else:
        N = len(loss_history)

    fig, axes = plt.subplots(N, 1, sharex=True)
    for ax, name in zip(axes.flat, loss_history.keys()):
        ax.plot(iteration, loss_history[name], color='blue')
        ax.legend([name], loc='upper right')
    plt.xlabel('# of iter')
    plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))

    if save_path is not None:
        plt.savefig(save_path)    

# plot dimension-wise KLD losses
def plot_KLD_losses(train_history, skip_N=100, plot_mean=True, plot_sum=True, save_path=None):
    iteration, loss_history = train_history
    if 'kld_loss_z' in loss_history:
        kld_loss_z = loss_history['kld_loss_z']
    else:
        return

    z_dim = len(kld_loss_z[0])
    kld_z_dict = {}
    for i in range(z_dim):
        kld_z_dict[str(i)] = []
    for step in kld_loss_z:
        for value, idx in zip(step, range(z_dim)):
            kld_z_dict[str(idx)].append(value)
    labels = []
    for i in range(len(kld_loss_z[0])):
        labels.append('z_' + str(i))
    
    if plot_mean:
        kld_loss_mean = []
        labels.append('mean')
    if plot_sum:
        kld_loss_sum = []
        labels.append('total')

    for value in kld_loss_z:
        if plot_mean:
            kld_loss_mean.append(value.mean())
        if plot_sum:
            kld_loss_sum.append(value.sum())

    fig, ax = plt.subplots(1,1)
    smooth_N = 100
    for kld_z in kld_z_dict.values():
        ax.plot(iteration[skip_N+smooth_N-1:], avg_smooth(kld_z[skip_N:], smooth_N), linewidth=1.0)
    if plot_mean:
        ax.plot(iteration[skip_N+smooth_N-1:], avg_smooth(kld_loss_mean[skip_N:], smooth_N), '--', color='k', linewidth=1.0, alpha=0.8)
    if plot_sum:
        ax.plot(iteration[skip_N+smooth_N-1:], avg_smooth(kld_loss_sum[skip_N:], smooth_N), '-', color='r', linewidth=1.0, alpha=0.8)
    plt.legend(labels) 
    plt.xlabel('# of iter')
    plt.title('KLD dimension-wise')

    if save_path is not None:
        plt.savefig(save_path)    

def exp_smooth(data, weight=0.6):  # Weight between 0 and 1
    last = data[0]  # First value in the plot (first timestep)
    smoothed = []
    for point in data:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)                        # Save it
        last = smoothed_val                                  # Anchor the last smoothed value

    return smoothed

def avg_smooth(data, N=10):  # N = number of points to be averaged
    data_np = np.array(data)
    smoothed = np.convolve(data_np, np.ones(N), 'valid') / N
    return smoothed