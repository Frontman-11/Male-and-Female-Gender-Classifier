# %% [code] {"execution":{"iopub.status.busy":"2024-08-10T21:08:19.561591Z","iopub.execute_input":"2024-08-10T21:08:19.562024Z","iopub.status.idle":"2024-08-10T21:08:33.930376Z","shell.execute_reply.started":"2024-08-10T21:08:19.561988Z","shell.execute_reply":"2024-08-10T21:08:33.929166Z"}}
import matplotlib.pyplot as plt
import cv2

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-08-10T21:00:21.395488Z","iopub.execute_input":"2024-08-10T21:00:21.395921Z","iopub.status.idle":"2024-08-10T21:00:21.404982Z","shell.execute_reply.started":"2024-08-10T21:00:21.395889Z","shell.execute_reply":"2024-08-10T21:00:21.403882Z"}}
def crop_square_resize(img, size=None, interpolation=cv2.INTER_AREA):
    if not size:
        print('image not cropped nor resized')
        return img
    h, w = img.shape[:2]
    min_size = np.amin([h,w])
    crop_img = img[int(h/2-min_size/2):int(h/2+min_size/2), int(w/2-min_size/2):int(w/2+min_size/2)]
    resized = cv2.resize(crop_img, (size, size), interpolation=interpolation)
    return resized


def read_plot_image(image_path, size=None, **kwargs):    
    flags = kwargs.get('flags', 1)
    image = cv2.imread(image_path, flags=flags)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = crop_square_resize(image, size)
    plt.imshow(image)
    plt.axis('off')
    plt.title(f'size: {image.shape}', loc='left')
    plt.show()
    
    
# %% [code] {"execution":{"iopub.status.busy":"2024-08-10T22:11:33.982071Z","iopub.execute_input":"2024-08-10T22:11:33.983400Z","iopub.status.idle":"2024-08-10T22:11:33.997201Z","shell.execute_reply.started":"2024-08-10T22:11:33.983360Z","shell.execute_reply":"2024-08-10T22:11:33.995865Z"}}
def plot_wrong_pred(dataset, y_true, preds, subplot_row_col=(20, 20), figsize=(40, 40), fname=None, take=-1):
    wrong_pred_idx = preds==y_true
    count = 0
    figure, axes = plt.subplots(subplot_row_col[0], subplot_row_col[1], figsize=figsize)
    axes = list(axes.flatten())
    ax_gen = iter(axes)
    n_title = 0
    
    for images, labels in valid_set.take(take):
        for image, label in zip(images, labels):
            if not wrong_pred_idx[count]:
                try:
                    ax = next(ax_gen)
                    ax.imshow(image)
                    ax.axis('off')
                    n_title += 1
                    ax.set_title(f'{n_title}: MALE' if label.numpy()==1 else f'{n_title}: FEMALE')
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
    plt.savefig(fname, dpi=300, format='png')
    plt.close()