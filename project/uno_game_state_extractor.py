import cv2
import numpy as np
import os
import math

class UnoGameStateExtractor:
    def __init__(self, template_dir="templates/"):
        """
        Initializes the pipeline, configuration thresholds, color ranges, 
        and loads template images.
        """
        # 1. Configuration
        self.percentage_white_pixels_to_be_white_background = 0.40
        self.threshold_template_matching_well = 0.90
        self.iou_threshold=0.3 
        
        self.height_player_box = 1622
        self.width_player_box = 900
                
        # 2. Define HSV Color Ranges for UNO Cards
        self.color_bounds = {
            'r': {
                'lower1': np.array([0, 70, 50]), 'upper1': np.array([10, 255, 255]),
                'lower2': np.array([170, 70, 50]), 'upper2': np.array([180, 255, 255])
            },
            'y': {'lower': np.array([20, 100, 100]), 'upper': np.array([35, 255, 255])},
            'g': {'lower': np.array([40, 50, 50]), 'upper': np.array([85, 255, 255])},
            'b': {'lower': np.array([90, 50, 50]), 'upper': np.array([125, 255, 255])},
            
            # STRICT BLACK (For Cards): Max Value lowered to 40. 
            # Must be almost pure, dark black.
            'black': {'lower': np.array([0, 0, 0]), 'upper': np.array([180, 255, 40])},
            
            # GRAY (For Token): Low saturation (0-50), allowing higher brightness (40-160).
            # The token detection combines this mask with the strict black mask above.
            'gray': {'lower': np.array([0, 0, 40]), 'upper': np.array([180, 50, 160])}
        }
        self.rois = {}
        self.target_symbols = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'skip', 'reverse', 'draw2', 'wild', 'draw4']
        self.templates = {'r': {}, 'y': {}, 'g': {}, 'b': {}} 
        self.templates = self._load_templates(template_dir)
        
        
    def _define_rois(self, img):
        """
        Defines the 5 bounding boxes based on precise pixel coordinates of the table setup.
        Coordinates are formatted as (x1, y1, x2, y2) ensuring x1 < x2 and y1 < y2.
        Origin (0,0) is at the top-left corner.
        """
        img_h, img_w = img.shape[:2] # Expected to be roughly 2662 (self.height_player_box) and 4000 (w)
        deported_vertical = int((img_h-self.height_player_box)/2) # around 520
        deport_horizontal= int((img_w-self.height_player_box)/2) # around 1000
        
        # Player 4 (Left)
        # User spec: start at y=520, x=0 | opposite at y=520+self.height_player_box, x=self.width_player_box
        p4_x1, p4_y1 = 0, deported_vertical
        p4_x2, p4_y2 = self.width_player_box, deported_vertical + self.height_player_box
        # Player 1 (Bottom)
        # User spec: start at y=img_h, x=1000 | opposite at y=img_h-self.width_player_box, x=1000+self.height_player_box
        # Reordered so y1 < y2
        p1_x1, p1_y1 = deport_horizontal, img_h - self.width_player_box
        p1_x2, p1_y2 = deport_horizontal + self.height_player_box, img_h
        # Player 3 (Top)
        # User spec: start at y=0, x=1000 | opposite at y=self.width_player_box, x=1000+self.height_player_box
        p3_x1, p3_y1 = deport_horizontal, 0
        p3_x2, p3_y2 = deport_horizontal + self.height_player_box, self.width_player_box
        # Player 2 (Right)
        # User spec: start at y=520, x=img_w-self.width_player_box | opposite at y=520+self.height_player_box, x=img_w
        p2_x1, p2_y1 = img_w - self.width_player_box, deported_vertical
        p2_x2, p2_y2 = img_w, deported_vertical + self.height_player_box
        # Center Zone (The space exactly between the inner edges of the player hands)
        # Bounded horizontally by Player 4 and Player 2
        # Bounded vertically by Player 3 and Player 1
        center_x1, center_y1 = p4_x2, p3_y2
        center_x2, center_y2 = p2_x1, p1_y1
        self.rois = {
            'p1_zone': (p1_x1, p1_y1, p1_x2, p1_y2),               # Bottom
            'p2_zone': (p2_x1, p2_y1, p2_x2, p2_y2),               # Right
            'p3_zone': (p3_x1, p3_y1, p3_x2, p3_y2),               # Top
            'p4_zone': (p4_x1, p4_y1, p4_x2, p4_y2),               # Left
            'center_zone': (center_x1, center_y1, center_x2, center_y2) # Middle
        }
        return
        
        
    def _load_templates(self, template_dir="training"):
        """
        Loads templates, converts to HSV to extract the white number, 
        and shares that purely binary template across ALL colors.
        """
        
        colors_dir = os.path.join(template_dir, 'templates_colors')
        
        if not os.path.exists(colors_dir):
            print(f"Error: Directory not found - {colors_dir}")
            return self.templates

        # HSV bounds for Pure White
        lower_white = np.array([0, 0, 180])
        upper_white = np.array([180, 60, 255])
        
        for symbol in self.target_symbols:
            for ext in ['.png', '.jpg', '.jpeg']:
                path = os.path.join(colors_dir, f"{symbol}{ext}")
                if os.path.exists(path):
                    img_color = cv2.imread(path)
                    if img_color is not None:
                        hsv = cv2.cvtColor(img_color, cv2.COLOR_BGR2HSV)
                        
                        # Create the crisp white-on-black binary mask
                        binary_template = cv2.inRange(hsv, lower_white, upper_white)
                        for color in self.templates.keys():
                            self.templates[color][symbol] = binary_template
                        break 
        return self.templates
    
    
    def _rotate_image(self, image, angle):
        """
        Rotates an image around its center without cropping the corners.
        Uses INTER_NEAREST to prevent anti-aliasing blur on binary images.
        """
        h, w = image.shape[:2]
        cx, cy = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))
        M[0, 2] += (new_w / 2) - cx
        M[1, 2] += (new_h / 2) - cy
        rotated = cv2.warpAffine(
            image, M, (new_w, new_h), 
            flags=cv2.INTER_NEAREST, 
            borderValue=0
        )
        return rotated


    def process_image(self, image_path, image_id):
        img = cv2.imread(image_path)
        if img is None: raise ValueError(f"Could not load image: {image_path}")
            
        self._define_rois(img)
        bg_type, processed_img = self._classify_and_preprocess_background(img)
        self.last_bg_type = bg_type 
        
        active_player, token_coords = self._detect_active_player(img, processed_img, bg_type, self.rois)
                
        # raw detections (Dictionaries with names and bounding boxes)
        raw_detections = {
            'p1': self._detect_cards_in_zone(processed_img, self.rois['p1_zone'], "p1"),
            'p2': self._detect_cards_in_zone(processed_img, self.rois['p2_zone'], "p2"),
            'p3': self._detect_cards_in_zone(processed_img, self.rois['p3_zone'], "p3"),
            'p4': self._detect_cards_in_zone(processed_img, self.rois['p4_zone'], "p4"),
            'center': self._detect_cards_in_zone(processed_img, self.rois['center_zone'], "center")
        }
        
        # TRIGGER VISUALIZATION 
        self.visualize_pipeline(processed_img, self.rois, token_coords, raw_detections)
        
        # FORMAT FOR KAGGLE
        player_cards_strings = {
            'p1': [d['name'] for d in raw_detections['p1']],
            'p2': [d['name'] for d in raw_detections['p2']],
            'p3': [d['name'] for d in raw_detections['p3']],
            'p4': [d['name'] for d in raw_detections['p4']]
        }
        
        center_card_string = raw_detections['center'][0]['name'] if raw_detections['center'] else "UNKNOWN"
        return self._format_output(image_id, center_card_string, active_player, player_cards_strings)


    def _non_max_suppression_iou(self, matches):
        """
        Industry-standard NMS using Intersection over Union (IoU).
        Removes overlapping bounding boxes, keeping only the highest scoring one.
        """
        if not matches: return []
        # Sort matches by score, highest to lowest
        matches.sort(key=lambda x: x['score'], reverse=True)
        keep = []
        for current_match in matches:
            box1 = current_match['bbox_rel'] # (x, y, w, h)
            overlap = False
            for kept_match in keep:
                box2 = kept_match['bbox_rel']               
                # Determine the coordinates of the intersection rectangle
                x_left = max(box1[0], box2[0])
                y_top = max(box1[1], box2[1])
                x_right = min(box1[0] + box1[2], box2[0] + box2[2])
                y_bottom = min(box1[1] + box1[3], box2[1] + box2[3])
                # Check if there is an actual intersection
                if x_right > x_left and y_bottom > y_top:
                    # Calculate areas
                    intersection_area = (x_right - x_left) * (y_bottom - y_top)
                    box1_area = box1[2] * box1[3]
                    box2_area = box2[2] * box2[3]                  
                    # Calculate Intersection over Union (IoU)
                    iou = intersection_area / float(box1_area + box2_area - intersection_area)                   
                    # If they overlap by more than our threshold (e.g., 30%), they are duplicates!
                    if iou > self.iou_threshold:
                        overlap = True
                        break                       
            if not overlap:
                keep.append(current_match)   
        return keep
    

    def _classify_and_preprocess_background(self, img):
        """Classifies background as 'white' or 'noise' and applies preprocessing."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, relaxed_white = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
        white_ratio = np.sum(relaxed_white == 255) / relaxed_white.size       
        bg_type = "white" if white_ratio > self.percentage_white_pixels_to_be_white_background else "noise"
        processed_img = img.copy()
        if bg_type == "noise":
            # Noise Preprocessing: Find white contours, close them, set rest to white
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            closed = cv2.morphologyEx(relaxed_white, cv2.MORPH_CLOSE, kernel)
            contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            mask = np.zeros_like(gray)
            cv2.drawContours(mask, contours, -1, 255, -1)
            # Set everything outside the contours to pure white
            processed_img[mask == 0] = [255, 255, 255]
        return bg_type, processed_img


    def _detect_active_player(self, img, processed_img, bg_type, rois):
        """Detects the token using contour geometry and maps it to the closest player zone."""
        hsv = cv2.cvtColor(processed_img, cv2.COLOR_BGR2HSV)
        token_center = None
        
        if bg_type == "white":
            # 1. Mask for Black AND Gray Rectangle
            mask_black = cv2.inRange(hsv, self.color_bounds['black']['lower'], self.color_bounds['black']['upper'])
            mask_gray = cv2.inRange(hsv, self.color_bounds['gray']['lower'], self.color_bounds['gray']['upper'])
            
            # Combine both masks
            mask = cv2.bitwise_or(mask_black, mask_gray)
            
            # Clean up the combined mask
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            target_area = 258 * 158 
            area_tolerance = 15000  
            best_contour = None
            closest_area_diff = float('inf')
            
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if abs(area - target_area) < area_tolerance:
                    rect = cv2.minAreaRect(cnt)
                    box_w, box_h = rect[1]
                    if box_w == 0 or box_h == 0: continue
                        
                    aspect_ratio = max(box_w, box_h) / min(box_w, box_h)
                    target_ratio = max(258, 158) / min(258, 158) 
                    
                    if abs(aspect_ratio - target_ratio) < 0.3:
                        if abs(area - target_area) < closest_area_diff:
                            closest_area_diff = abs(area - target_area)
                            best_contour = rect 
            
            if best_contour:
                token_center = (int(best_contour[0][0]), int(best_contour[0][1]))
                
        else:
            # 2. Mask for Yellow Circle (Noise Background)
            mask = cv2.inRange(hsv, self.color_bounds['y']['lower'], self.color_bounds['y']['upper'])
            
            # Use an elliptical kernel to clean up a circular mask
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            best_circularity = 0
            best_cnt = None
            
            for cnt in contours:
                area = cv2.contourArea(cnt)
                
                # Ignore tiny yellow specs on the table (noise)
                if area < 1000: 
                    continue
                    
                perimeter = cv2.arcLength(cnt, True)
                if perimeter == 0:
                    continue
                    
                # Calculate Circularity
                circularity = 4 * math.pi * (area / (perimeter * perimeter))
                
                # If it looks like a circle (> 0.75) and is the most perfect circle we've found
                if circularity > 0.60 and circularity > best_circularity:
                    best_circularity = circularity
                    best_cnt = cnt
                    
            if best_cnt is not None:
                # Find the center of the circle using image moments
                M = cv2.moments(best_cnt)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    token_center = (cX, cY)

        # Fallback if token is entirely missing
        if token_center is None:
            token_center = (img.shape[1]//2, img.shape[0]//2) 
            print("Warning: Active player token not detected, defaulting to image center.")
        
        # 4. Find closest ROI to determine the active player
        closest_player = "p1"
        min_dist = float('inf')
        
        tx, ty = token_center
        
        for player in ['p1', 'p2', 'p3', 'p4']:
            x1, y1, x2, y2 = rois[f"{player}_zone"]
            dx = max(x1 - tx, 0, tx - x2)
            dy = max(y1 - ty, 0, ty - y2)
            dist = math.hypot(dx, dy)
            
            if dist < min_dist:
                min_dist = dist
                closest_player = player
                
        return closest_player, token_center


    def _detect_cards_in_zone(self, img, roi, zone_name="Unknown"):
        """
        Isolates white numbers using HSV saturation, masks by color, 
        and matches rotated templates robustly against perspective skew.
        """
        x1, y1, x2, y2 = roi
        zone_img = img[y1:y2, x1:x2]
        
        if zone_img.size == 0:
            return []
            
        hsv_zone = cv2.cvtColor(zone_img, cv2.COLOR_BGR2HSV)
        
        # 1. Isolate ALL white numbers in the zone using the HSV trick
        lower_white = np.array([0, 0, 180])
        upper_white = np.array([180, 60, 255])
        all_white_numbers = cv2.inRange(hsv_zone, lower_white, upper_white)
        
        detected_cards = []
        
        for color, bounds in self.color_bounds.items():
            if color == 'gray': continue 
            if not self.templates.get(color): continue
            
            # 2. Get the specific color mask
            if color == 'r':
                mask1 = cv2.inRange(hsv_zone, bounds['lower1'], bounds['upper1'])
                mask2 = cv2.inRange(hsv_zone, bounds['lower2'], bounds['upper2'])
                color_mask = cv2.bitwise_or(mask1, mask2)
            else:
                color_mask = cv2.inRange(hsv_zone, bounds['lower'], bounds['upper'])
            
            # 3. Dilate the color mask to cover the white numbers
            kernel = np.ones((15, 15), np.uint8)
            dilated_color = cv2.dilate(color_mask, kernel, iterations=2)
            
            # 4. Extract ONLY the white numbers belonging to this color
            isolated_binary_numbers = cv2.bitwise_and(all_white_numbers, dilated_color)
            
            # List to hold ALL symbols for this specific color
            all_color_matches = []
            
            # 5. Template Matching
            for symbol, base_template in self.templates[color].items():
                for angle in range(0, 360, 15):
                    rotated_template = self._rotate_image(base_template, angle)
                    h_rot, w_rot = rotated_template.shape[:2]
                    
                    res = cv2.matchTemplate(isolated_binary_numbers, rotated_template, cv2.TM_CCOEFF_NORMED)
                    
                    loc = np.where(res >= self.threshold_template_matching_well) 
                    
                    for pt in zip(*loc[::-1]):
                        center_pt = (pt[0] + w_rot // 2, pt[1] + h_rot // 2)
                        
                        all_color_matches.append({
                            'symbol': symbol,
                            'center': center_pt, 
                            'score': res[pt[1], pt[0]],
                            'bbox_rel': (pt[0], pt[1], w_rot, h_rot) 
                        })
            
            # STAGE 1: Apply NMS across ALL symbols of this color.
            nms_color_matches = self._non_max_suppression_iou(all_color_matches)
            
            # STAGE 2: Group the surviving cards by symbol and apply the Top 4 limit (for 1 symbol)
            symbol_groups = {}
            for match in nms_color_matches:
                sym = match['symbol']
                if sym not in symbol_groups:
                    symbol_groups[sym] = []
                symbol_groups[sym].append(match)
                
            for sym, matches in symbol_groups.items():
                # Keep maximum 4 of THIS specific symbol
                top_4_for_symbol = matches[:4]
                for match in top_4_for_symbol:
                    rel_x, rel_y, w, h = match['bbox_rel']
                    detected_cards.append({
                        'name': f"{color}_{sym}",
                        'bbox': (x1 + rel_x, y1 + rel_y, w, h),
                        'score': match['score']
                    })
                        
        return detected_cards
    
   

    def _format_output(self, image_id, center_card, active_player, player_cards):
        """Formats the data specifically for the Kaggle CSV constraints."""
        formatted_row = [image_id, center_card, active_player]
        
        for p in ['p1', 'p2', 'p3', 'p4']:
            cards = player_cards[p]
            if not cards:
                formatted_row.append("EMPTY")
            else:
                formatted_row.append(";".join(cards))
                
        return formatted_row
    
    
    def visualize_pipeline(self, cleaned_img, rois, token_coords, all_detections):
        """Draws the ROIs, Active Player Token, and Detected Templates on the image."""
        viz_img = cleaned_img.copy()
        
        COLOR_TOKEN = (0, 0, 255)     # Red
        COLOR_CARD = (0, 255, 0)      # Green
        COLOR_ROI = (255, 255, 0)     # Cyan (for debugging zones)

        # 1. Draw Player Zones (ROIs)
        for name, (x1, y1, x2, y2) in rois.items():
            cv2.rectangle(viz_img, (x1, y1), (x2, y2), COLOR_ROI, 2)
            # Add the name of the zone so you know which is which
            cv2.putText(viz_img, name.upper(), (x1 + 10, y1 + 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_ROI, 2)

        # 2. Draw Active Player Token
        if token_coords:
            tx, ty = token_coords
            cv2.rectangle(viz_img, (tx - 20, ty - 20), (tx + 20, ty + 20), COLOR_TOKEN, 4)
            cv2.putText(viz_img, "TOKEN", (tx - 20, ty - 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_TOKEN, 2)

        # 3. Draw Detected Templates
        for zone, detections in all_detections.items():
            for det in detections:
                x, y, w, h = det['bbox']
                card_label = f"{det['name']} ({det['score']:.2f})"
                
                # Draw Box
                cv2.rectangle(viz_img, (x, y), (x + w, y + h), COLOR_CARD, 2)
                # Draw Label with a black background for readability
                cv2.rectangle(viz_img, (x, y - 20), (x + len(card_label)*12, y), (0,0,0), -1)
                cv2.putText(viz_img, card_label, (x + 2, y - 5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_CARD, 2)

        cv2.namedWindow("Final Detections", cv2.WINDOW_NORMAL)
        cv2.imshow("Final Detections", viz_img)
        cv2.waitKey(0) 
        cv2.destroyAllWindows()