from ab.nn.loader.coco_ import Detection, Segmentation, Caption,textimage

def loader(transform_fn, task):
    if task == 'obj-detection':
        f = Detection
    elif task == 'img-segmentation':
        f = Segmentation
    elif task == 'img-captioning':
        f = Caption
    elif task == 'text2image':
        f = textimage
    else:
        raise Exception(f"The task '{task}' is not implemented for COCO dataset.")

    return f.loader(transform_fn, task)
