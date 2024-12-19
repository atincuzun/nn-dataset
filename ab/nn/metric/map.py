def compute(predictions, targets):
    """
    Compute mean Average Precision for object detection predictions
    
    Args:
        predictions: List[Dict] where each dict contains:
            'boxes': tensor (N, 4)
            'scores': tensor (N,)
            'labels': tensor (N,)
        targets: List[Dict] where each dict contains:
            'boxes': tensor (M, 4)
            'labels': tensor (M,)
    
    Returns:
        tuple: (map_score, 1) - MAP score and count=1 to match metric interface
    """
    from pycocotools.cocoeval import COCOeval
    from pycocotools.coco import COCO
    
    # Initialize COCO format structures
    coco_gt = COCO()
    coco_dt = COCO()
    
    # Prepare data structures
    images = []
    annotations = []
    pred_annotations = []
    img_id = 1
    ann_id = 1
    
    # Process each image
    for target, pred in zip(targets, predictions):
        # Add image info
        images.append({
            'id': img_id,
            'height': 100,  # placeholder
            'width': 100,   # placeholder
        })
        
        # Process ground truth
        gt_boxes = target['boxes']
        gt_labels = target['labels']
        for box, label in zip(gt_boxes, gt_labels):
            x1, y1, x2, y2 = box.tolist()
            w = x2 - x1
            h = y2 - y1
            annotations.append({
                'id': ann_id,
                'image_id': img_id,
                'category_id': int(label),
                'bbox': [x1, y1, w, h],
                'area': w * h,
                'iscrowd': 0
            })
            ann_id += 1
        
        # Process predictions
        pred_boxes = pred['boxes']
        pred_scores = pred['scores']
        pred_labels = pred['labels']
        
        for box, score, label in zip(pred_boxes, pred_scores, pred_labels):
            x1, y1, x2, y2 = box.tolist()
            w = x2 - x1
            h = y2 - y1
            pred_annotations.append({
                'image_id': img_id,
                'category_id': int(label),
                'bbox': [x1, y1, w, h],
                'score': float(score)
            })
        
        img_id += 1
    
    # Create datasets
    num_classes = max(max(ann['category_id'] for ann in annotations), 
                     max(pred['category_id'] for pred in pred_annotations)) + 1
    
    dataset = {
        'images': images,
        'annotations': annotations,
        'categories': [{'id': i} for i in range(num_classes)]
    }
    coco_gt.dataset = dataset
    coco_gt.createIndex()
    
    coco_dt.dataset = {
        'images': images,
        'annotations': pred_annotations,
        'categories': [{'id': i} for i in range(num_classes)]
    }
    coco_dt.createIndex()
    
    # Evaluate
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    
    # Get mAP score (IoU=0.50:0.95)
    map_score = float(coco_eval.stats[0])
    
    return map_score, 1  # Return score and count=1 to match metric interface
