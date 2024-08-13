{"metadata":{"kernelspec":{"language":"python","display_name":"Python 3","name":"python3"},"language_info":{"name":"python","version":"3.10.13","mimetype":"text/x-python","codemirror_mode":{"name":"ipython","version":3},"pygments_lexer":"ipython3","nbconvert_exporter":"python","file_extension":".py"},"kaggle":{"accelerator":"none","dataSources":[],"dockerImageVersionId":30746,"isInternetEnabled":true,"language":"python","sourceType":"script","isGpuEnabled":false}},"nbformat_minor":4,"nbformat":4,"cells":[{"cell_type":"code","source":"# %% [code] {\"jupyter\":{\"outputs_hidden\":false},\"execution\":{\"iopub.status.busy\":\"2024-08-13T00:42:25.645337Z\",\"iopub.execute_input\":\"2024-08-13T00:42:25.645766Z\",\"iopub.status.idle\":\"2024-08-13T00:42:25.653973Z\",\"shell.execute_reply.started\":\"2024-08-13T00:42:25.645735Z\",\"shell.execute_reply\":\"2024-08-13T00:42:25.652425Z\"}}\nimport tensorflow as tf\n\n\ndef return_pred_and_label(dataset, count=-1):\n    predictions = tf.constant([], dtype=tf.int64)\n    y_true = tf.constant([], dtype=tf.int64)\n\n    for image, label in dataset.take(count, model=None):\n        y_pred = tf.cast((model.predict(image, verbose=0) >= 0.5).reshape(-1), dtype=tf.int64)\n        predictions = tf.concat([predictions, y_pred], -1)\n        y_true = tf.concat([y_true, label], -1)\n    return {'y_true':y_true, 'y_pred':predictions}\n","metadata":{"_uuid":"e3105482-2f2f-4d37-b22c-e8c31a2b3b1a","_cell_guid":"45f28b92-b54a-47fe-a246-2897aedb6a78","collapsed":false,"jupyter":{"outputs_hidden":false},"trusted":true},"execution_count":null,"outputs":[]}]}