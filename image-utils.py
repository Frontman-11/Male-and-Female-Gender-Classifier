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