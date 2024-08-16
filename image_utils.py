import matplotlib.pyplot as plt
import cv2
import numpy as np

def crop_square_resize(img, size=None, interpolation=cv2.INTER_AREA):
    """
    Crops the central square region from an image and resizes it to the specified size.

    Args:
        img (np.ndarray): The input image.
        size (int, optional): The target size for the resized image. If None, no resizing is done.
        interpolation (int, optional): Interpolation method for resizing. Default is cv2.INTER_AREA.

    Returns:
        np.ndarray: The cropped and resized image.
    """
    if not size:
        print('Image not cropped nor resized')
        return img
    h, w = img.shape[:2]
    min_size = np.amin([h, w])
    crop_img = img[int(h / 2 - min_size / 2):int(h / 2 + min_size / 2), int(w / 2 - min_size / 2):int(w / 2 + min_size / 2)]
    resized = cv2.resize(crop_img, (size, size), interpolation=interpolation)
    return resized


def read_plot_image(image_path, size=None, **kwargs):
    """
    Reads an image from a file, processes it, and plots it using matplotlib.

    Args:
        image_path (str): Path to the image file.
        size (int, optional): The target size for the resized image. If None, no resizing is done.
        **kwargs: Additional arguments for cv2.imread (e.g., flags).

    Returns:
        None
    """
    flags = kwargs.get('flags', 1)
    image = cv2.imread(image_path, flags=flags)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = crop_square_resize(image, size)
    plt.imshow(image)
    plt.axis('off')
    plt.title(f'Size: {image.shape}', loc='left')
    plt.show()


def plot_wrong_pred(dataset,
                    y_true,
                    preds,
                    subplot_row_col=(20, 20),
                    figsize=(40, 40), 
                    fname=None,
                    take=-1,
                    show=False
                   ):
    """
    Plots images with wrong predictions.

    Args:
        dataset (tf.data.Dataset): Dataset containing images and labels.
        y_true (np.ndarray): Array of true labels.
        preds (np.ndarray): Array of predicted labels.
        subplot_row_col (tuple, optional): Number of rows and columns for subplots. Default is (20, 20).
        figsize (tuple, optional): Size of the figure. Default is (40, 40).
        fname (str, optional): Path to save the plot image. If None, the image is not saved.
        take (int, optional): Number of samples to take from the dataset. Default is -1 (take all).
        show(bool, optional): Whether to display the plot. Default is False.

    Returns:
        None
    """
    wrong_pred_idx = preds != y_true
    count = 0
    figure, axes = plt.subplots(subplot_row_col[0], subplot_row_col[1], figsize=figsize)
    axes = list(axes.flatten())
    ax_gen = iter(axes)
    n_title = 0
    
    for images, labels in dataset.take(take):
        for image, label in zip(images, labels):
            if wrong_pred_idx[count]:
                try:
                    ax = next(ax_gen)
                    ax.imshow(image)
                    ax.axis('off')
                    n_title += 1
                    ax.set_title(f'{n_title}: MALE' if label.numpy() == 1 else f'{n_title}: FEMALE')
                except StopIteration:
                    print('Out of axis')
                    break
            count += 1
            
    while True:
        try:
            ax = next(ax_gen)
            ax.axis('off')
        except StopIteration:
            break
            
    plt.tight_layout()
    if fname:
        plt.savefig(fname, dpi=300, format='png')
    if show:
        plt.show()
    plt.close()