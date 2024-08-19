import tensorflow as tf


def stack_filepaths(paths: list, max_length=None):
    """
    Stack file paths into a single list, repeating entries to match the maximum length.

    Args:
        paths (list of list of str): List of lists containing file paths.
        max_length (int, optional): Maximum length to stack file paths. Defaults to None.

    Returns:
        list of str: Stacked file paths.
    """
    length = max_length or max(len(path) for path in paths)
    
    filepaths = []
    for i in range(length):
        for path in paths:
            try:
                filepaths.extend([path[i]])
            except IndexError:
                continue
    return filepaths
