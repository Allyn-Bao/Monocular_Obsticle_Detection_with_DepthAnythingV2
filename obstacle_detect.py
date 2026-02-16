import cv2
import torch

from depth_anything_v2.dpt import DepthAnythingV2
import numpy as np
import os
import math


class ObstacleDetector:

    def __init__(self, encoder='vits', dist_threshold=2, mask_threshold=1.6, foreground_threshold=3, background_threshold=0.2, debug=False, write_img=False):
        DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

        model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
        }

        encoder = 'vits' # or 'vits', 'vitb', 'vitg'

        self.model = DepthAnythingV2(**model_configs[encoder])
        self.model.load_state_dict(torch.load(f'checkpoints/depth_anything_v2_{encoder}.pth', map_location='cpu'))
        self.model = self.model.to(DEVICE).eval()

        self.dist_threshold = dist_threshold # used in absolute depth thresholding
        self.mask_threshold = mask_threshold # used in masking normalzied depth for bbox extraction

        self.foreground_threshold = foreground_threshold # used in finding foreground row
        self.background_threshold = background_threshold # used in finding background row

        self.debug = debug
        self.write_img = write_img
        if self.write_img:
            os.makedirs("./debug", exist_ok=True)

    def detect_obstacles(self, img_path):
        """
        Docstring for detect_obstacles
        
        :param img_path: Description
        :return: list[bbox], bbox: (x, y, w, h)
        """

        print("[INFO] Processing image for obstacle detection... debug mode:", self.debug)

        # preprocess image
        img = cv2.imread(img_path)
        img = img = cv2.resize(img, (480, 360), interpolation=cv2.INTER_AREA)
        depth = self.model.infer_image(img)
        og_depth = depth.copy() # for debugging visualization
        
        # crop center
        crop_x = 100
        crop_y_top = 150
        crop_y_bottom = 100
        # img = img[crop_y_top:360-crop_y_bottom, crop_x:480-crop_x]
        # depth = depth[crop_y_top:360-crop_y_bottom, crop_x:480-crop_x]

        print(f"[INFO] Depth map shape: {depth.shape}, min: {np.nanmin(depth)}, max: {np.nanmax(depth)}")

        best_angle = self.find_best_orientation(depth)
        print(f"[INFO] Best orientation angle: {best_angle} degrees")
        depth = self.rotate(depth, angle_deg=best_angle, is_depth=True)
        rotated_depth = depth.copy() # for debugging visualization

        # trivial case, approaching wall
        cropped_depth = depth[crop_y_top:360-crop_y_bottom, crop_x:480-crop_x]
        median_depth = np.median(cropped_depth)
        if median_depth > self.dist_threshold:
            print(f"[INFO] Median depth {median_depth:.2f} below threshold, likely approaching wall. Returning full image bbox.")
            return [(0, 0, img.shape[1], img.shape[0])]
        
        # find foreground
        foreground_row = 360
        for row in range(depth.shape[0]):
            if np.median(depth[row, :]) > self.foreground_threshold:
                foreground_row = row
                break
        print(f"[INFO] Detected foreground row at: {foreground_row}")

        # find background
        background_row = 0
        for row in range(depth.shape[0]):
            if np.mean(depth[row, :]) > self.background_threshold:
                background_row = row
                break
        print(f"[INFO] Detected background row at: {background_row}")

        depth = depth[background_row:foreground_row, :]
        
        # row normalization to find obsticle anomalies
        normalized_depth = []
        for row in range(depth.shape[0]):
            mean = np.nanmean(depth[row, :])
            std = np.nanstd(depth[row, :])
            normalized_depth.append((depth[row, :] - mean) / std + 1e-8)
        normalized_depth = np.array(normalized_depth)

        print(f"[INFO] Normalized depth map shape: {normalized_depth.shape}, min: {np.nanmin(normalized_depth)}, max: {np.nanmax(normalized_depth)}")
        Norm_MAX = np.nanmax(normalized_depth)

        bboxes, mask = self.bboxes_from_depth(normalized_depth, threshold=self.dist_threshold, min_area=10, do_morph=False, morph_ksize=1)

        updated_bboxes = [] # restore to original image coordinates 
        for bbox in bboxes:
            x, y, w, h = bbox
            updated_bbox = (x, y + background_row, w, h)
            # resize bbox back to original image coordinates
            scale_x = img.shape[1] / depth.shape[1]
            scale_y = img.shape[0] / depth.shape[0]
            updated_bbox = (int(updated_bbox[0] * scale_x), int(updated_bbox[1] * scale_y), int(updated_bbox[2] * scale_x), int(updated_bbox[3] * scale_y))
            updated_bboxes.append(updated_bbox)

        
        if self.debug:
            print("[DEBUG] Debug Mode On, visualizing depth and bboxes...")
            self.visualize_img(img)
            self.visualize_depth(og_depth, label="estimated depth map")
            self.visualize_depth(rotated_depth, label="rotated depth map based on foregrand gradient")
            self.visualize_depth(normalized_depth, label="cropped & row-normalized depth map")
            self.visualize_depth(mask, label="binary mask from thresholding")
            self.visualize_bboxes(img, updated_bboxes)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        return bboxes
    

    def find_best_orientation(self, depth):
        angles = [-10, -5, -2.5, -1, 0, 1, 2.5, 5, 10]
        # find foreground boundary
        foreground_row = depth.shape[0] - 1
        for row in range(depth.shape[0]):
            if np.median(depth[row, :]) > 3:
                foreground_row = row
                break

        min_loss = float('inf')
        for angle in angles:
            rotated_depth = self.rotate(depth, angle_deg=angle, is_depth=True)
            for row in range(foreground_row, rotated_depth.shape[0], 1):
                # minimize variance in each row (perfect horizontal foreground would have zero variance)
                loss = np.mean((rotated_depth[row, :] - np.mean(rotated_depth[row, :])) ** 2)
                if loss < min_loss:
                    min_loss = loss
                    best_angle = angle
        
        return best_angle

    def rotate(self, img, angle_deg, is_depth=False):
        """
        (This is written by chatGPT)
        Rotate image while keeping same output size and NO blank pixels.
        Depth maps are rotated via geometric reprojection (no interpolation artifacts).

        Args:
            img: HxW (depth) or HxWx3 (RGB)
            angle_deg: rotation angle (positive = CCW)
            is_depth: True if img is a depth map
        Returns:
            rotated_img same size as input
        """

        h, w = img.shape[:2]
        angle = math.radians(angle_deg)

        # -------------------------------------------------
        # 1) rotate into expanded canvas
        # -------------------------------------------------
        diag = int(np.sqrt(h*h + w*w))
        pad_y = (diag - h) // 2
        pad_x = (diag - w) // 2

        if img.ndim == 3:
            canvas = np.zeros((diag, diag, img.shape[2]), dtype=img.dtype)
        else:
            canvas = np.zeros((diag, diag), dtype=img.dtype)

        canvas[pad_y:pad_y+h, pad_x:pad_x+w] = img
        center = (diag/2, diag/2)

        # -------------------------------------------------
        # 2) rotate
        # -------------------------------------------------
        if not is_depth:
            # Normal image rotation (interpolated)
            M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
            rotated = cv2.warpAffine(canvas, M, (diag, diag),
                                    flags=cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_REPLICATE)

        else:
            # Depth rotation via inverse coordinate mapping
            yy, xx = np.indices((diag, diag))
            cx, cy = center

            x = xx - cx
            y = yy - cy

            cosA = math.cos(angle)
            sinA = math.sin(angle)

            src_x =  cosA*x + sinA*y + cx
            src_y = -sinA*x + cosA*y + cy

            src_x = np.round(src_x).astype(np.int32)
            src_y = np.round(src_y).astype(np.int32)

            valid = (src_x>=0)&(src_x<diag)&(src_y>=0)&(src_y<diag)

            rotated = np.zeros_like(canvas)
            rotated[valid] = canvas[src_y[valid], src_x[valid]]

        # -------------------------------------------------
        # 3) compute largest valid rectangle
        # -------------------------------------------------
        sin_a = abs(math.sin(angle))
        cos_a = abs(math.cos(angle))

        wr = int(w*cos_a - h*sin_a)
        hr = int(h*cos_a - w*sin_a)

        wr = max(1, wr)
        hr = max(1, hr)

        cx, cy = int(center[0]), int(center[1])
        crop = rotated[cy-hr//2:cy+hr//2, cx-wr//2:cx+wr//2]

        # -------------------------------------------------
        # 4) resize back to original resolution
        # -------------------------------------------------
        interp = cv2.INTER_NEAREST if is_depth else cv2.INTER_LINEAR
        final = cv2.resize(crop, (w, h), interpolation=interp)

        return final
    
    def bboxes_from_depth(self, depth: np.ndarray,
                      threshold: float,
                      min_area: int = 5,
                      connectivity: int = 4,
                      do_morph: bool = True,
                      morph_ksize: int = 3):
        """
        (This is writen by chatGPT)
        depth: HxW float32/float64 depth map (normalized or not)
        threshold: keep pixels with depth <= threshold (near objects)
        returns: list of (x, y, w, h) bboxes
        """
        assert depth.ndim == 2, "depth must be HxW"

        # 1) Binary mask: within threshold
        mask = (depth >= threshold).astype(np.uint8) * 255  # 0 or 255

        # 2) Optional cleanup (reduces speckles & fills small holes)
        if do_morph:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_ksize, morph_ksize))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k, iterations=1)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=1)

        # 3) Connected components with stats
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            mask, connectivity=connectivity
        )

        # stats[i] = [x, y, w, h, area] for component i
        # label 0 is background
        bboxes = []
        for i in range(1, num_labels):
            x, y, w, h, area = stats[i]
            if area >= min_area:
                bboxes.append((int(x), int(y), int(w), int(h)))

        return bboxes, mask
    
    def visualize_img(self, img):
        if self.write_img:
            cv2.imwrite("./debug/debug_image.png", img)
        else:
            cv2.imshow("image", img)
    
    def visualize_bboxes(self, img, bboxes):
        for (x, y, w, h) in bboxes:
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        if self.write_img:
            cv2.imwrite("./debug/debug_bboxes.png", img)
        else:
            cv2.imshow("bboxes", img)
    
    def visualize_depth(self, depth, label="depth"):
        colored_depth = self.depth_to_colormap(depth)
        if self.write_img:
            cv2.imwrite(f"./debug/debug_{label}.png", colored_depth)
        else:
            cv2.imshow(label, colored_depth)

    def depth_to_colormap(self, depth_hw: np.ndarray, colormap=cv2.COLORMAP_TURBO) -> np.ndarray:
        d = depth_hw.astype(np.float32)
        d = d - np.nanmin(d)
        denom = (np.nanmax(d) - np.nanmin(d))
        if denom < 1e-8:
            denom = 1.0
        d = (d / denom * 255.0).clip(0, 255).astype(np.uint8)
        colored = cv2.applyColorMap(d, colormap)  # BGR uint8
        return colored
    

if __name__ == "__main__":
    detector = ObstacleDetector(debug=True, write_img=True)
    test_img_path = "/Users/allynbao/Documents/ml/ORCA_computer_vision/ORCA_Computer_Vision/datasets/front-cam/run_24_front/img_027.jpg"
    print("Process image")
    bboxes = detector.detect_obstacles(test_img_path)
    print(f"Detected bboxes: {bboxes}")
    