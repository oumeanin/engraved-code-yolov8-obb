"""Final-paper condition names and the settings recorded in the source code."""

COMMON = dict(
    epochs=100,
    close_mosaic=0,
    imgsz=1280,
    batch=4,
    patience=15,
    device=0,
    seed=0,
    deterministic=True,
    workers=0,
)

AUG_OFF = dict(
    hsv_h=0.0, hsv_s=0.0, hsv_v=0.0,
    degrees=0.0, translate=0.0, scale=0.0, shear=0.0, perspective=0.0,
    flipud=0.0, fliplr=0.0,
    mosaic=0.0, mixup=0.0, erasing=0.0,
)

AUG_ON = dict(
    hsv_h=0.0, hsv_s=0.7, hsv_v=0.4,
    degrees=10.0, translate=0.05, scale=0.1, shear=0.0, perspective=0.0,
    flipud=0.0, fliplr=0.0,
    mosaic=0.0, mixup=0.0, erasing=0.0,
)

EXPERIMENTS = {
    "a": {"legacy": "c", "oversample": False, "augmentation": False},
    "b": {"legacy": "d", "oversample": True, "augmentation": False},
    "c": {"legacy": "e", "oversample": False, "augmentation": True},
    "d": {"legacy": "f", "oversample": True, "augmentation": True},
}

CLASS_MULTIPLIERS = {
    "ZZ": 10, "TT": 10, "SS": 10, "11": 10, "VV": 10,
    "YY": 7, "55": 7,
    "NN": 5, "DD": 5, "GG": 5, "MM": 5,
    "WW": 5, "PP": 5, "FF": 5,
}

INFERENCE = dict(imgsz=1280, conf=0.30, iou=0.20)
