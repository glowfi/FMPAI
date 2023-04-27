import face_detection

detector = face_detection.build_detector(
    "RetinaNetResNet50", confidence_threshold=0.5, nms_iou_threshold=0.3
)
