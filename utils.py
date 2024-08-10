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
    
    
    
def return_pred_and_label(dataset, count=-1):
    predictions = tf.constant([], dtype=tf.int64)
    y_true = tf.constant([], dtype=tf.int64)

    for image, label in dataset.take(count):
        y_pred = tf.cast((model.predict(image, verbose=0) >= 0.5).reshape(-1), dtype=tf.int64)
        predictions = tf.concat([predictions, y_pred], -1)
        y_true = tf.concat([y_true, label], -1)
    return {'y_true':y_true, 'y_pred':predictions}