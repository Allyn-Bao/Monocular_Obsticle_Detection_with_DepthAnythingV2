# Monocular Obsticle Detection with DepthAnything V2

This repository implements an image processing pipeline to detect and localize arbitary objects from monocular camera feed in road or water envirnments.
The repository is forked from DepthAnything V2: https://github.com/DepthAnything/Depth-Anything-V2

## To run:
```
from obsticle_detect import ObstacleDetector

detector = ObstacleDetector(debug=True)
test_img_path = "/path/to/img"
bboxes = detector.detect_obstacles(test_img_path)
```