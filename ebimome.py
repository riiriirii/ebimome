import time
import dataclasses
import mediapipe as mp
import cv2 as cv
import pygame
class MpHandLandmarker:
    def __init__(self):
        self.options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path="hand_landmarker.task"),
            running_mode=mp.tasks.vision.RunningMode.LIVE_STREAM,
            num_hands=2,
            result_callback=self.get_result
        )
        self.result = None
        self.timestamp = None
    def __enter__(self):
        self.hand_landmarker = mp.tasks.vision.HandLandmarker.create_from_options(self.options)
        return self
    def __exit__(self, exception_type, exception_value, traceback):
        self.hand_landmarker.close()
    def get_result(self, result, output_image, timestamp_ms):
        self.result = result
        self.timestamp = timestamp_ms
    def detect(self, frame):
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        self.hand_landmarker.detect_async(mp_image, int(time.time() * 1000))

@dataclasses.dataclass
class LandmarkCoordinates:
    hand: str
    x_min: int
    y_min: int
    x_max: int
    y_max: int
    x_nine: int
    y_nine: int

class LandmarkResultExtractor:
    def __init__(self, frame_width, frame_height, hand_landmerker_result):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.handedness_list = hand_landmerker_result.handedness
        self.hand_landmarks_list = hand_landmerker_result.hand_landmarks
        self.extracted_landmark_list = []
        self.hands_set = set()
    def get_coordinates(self):
        for idx, hand_landmarks in enumerate(self.hand_landmarks_list):
            x_coordinates = [landmark.x for landmark in hand_landmarks]
            y_coordinates = [landmark.y for landmark in hand_landmarks]
            landmark_coordinates = LandmarkCoordinates(
                self.handedness_list[idx][0].category_name,
                int(min(x_coordinates) * self.frame_width),
                int(min(y_coordinates) * self.frame_height),
                int(max(x_coordinates) * self.frame_width),
                int(max(y_coordinates) * self.frame_height),
                int(x_coordinates[9] * self.frame_width),
                int(y_coordinates[9] * self.frame_height)
                )
            self.extracted_landmark_list.append(landmark_coordinates)
        return self.extracted_landmark_list
    def is_both_hands(self):
        self.hands_set = {handedness[0].category_name for handedness in self.handedness_list}
        if len(self.hands_set) == 2:
            return True
        return False

class EbiMomiComboTracker:
    def __init__(self, frame_width, frame_height, touch_margin=20, touch_time_limit=1000, combo_threshold=5):
        self.x = frame_width - frame_width // 3
        self.y_top = frame_height // 2
        self.y_bottom = frame_height - frame_height // 3
        self.touch_margin = touch_margin
        self.touch_time_limit = touch_time_limit
        self.combo_threshold = combo_threshold
        self.reset_touch_state()
    def reset_touch_state(self):
        self.previous_side = None
        self.last_count_time = 0
        self.combo_num = 0
    def is_combo(self):
        if self.combo_num >= self.combo_threshold:
            return True
        return False
    def is_moving(self, current_time):
        if current_time - self.last_count_time > self.touch_time_limit:
            self.last_count_time = current_time
            return False
        return True
    def get_touching_side(self, extracted_landmark_list):
        for landmark_coordinates in extracted_landmark_list:
            if abs(landmark_coordinates.x_nine - self.x) <= self.touch_margin:
                if abs(landmark_coordinates.y_nine - self.y_top) <= self.touch_margin:
                    return 'top'
                elif abs(landmark_coordinates.y_nine - self.y_bottom) <= self.touch_margin:
                    return 'bottom'
        return None
    def try_count_combo(self, current_side, current_time):
        if self.previous_side == current_side:
            return False
        if self.previous_side != None:
            self.combo_num += 1
        self.previous_side = current_side
        self.last_count_time = current_time
        return True

class CvVideoCap:
    def __init__(self):
        self.cap = cv.VideoCapture(0)
    def __enter__(self):
        if not self.cap.isOpened():
            raise RuntimeError("Cannot open camera")
        self.width = self.cap.get(cv.CAP_PROP_FRAME_WIDTH)
        self.height = self.cap.get(cv.CAP_PROP_FRAME_HEIGHT)
        return self
    def __exit__(self, exception_type, exception_value, traceback):
        self.cap.release()
        cv.destroyAllWindows()
    def read_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("Can't receive frame (stream end?). Exiting ...")
        frame = cv.flip(frame, 1)
        return frame
    def show_frame(self, frame):
        if frame is not None:
            cv.imshow("frame", frame)
    def wait(self, key='q'):
        if cv.waitKey(1) == ord(key):
            return False
        return True

class ImageRendere:
    def __init__(self, frame_width, frame_height, right_hand_img_path="right_hand.png", left_hand_img_path="left_hand.png", ebi_img_path="ebi.png"):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.right_hand_img = cv.imread(right_hand_img_path, cv.IMREAD_UNCHANGED)
        self.left_hand_img = cv.imread(left_hand_img_path, cv.IMREAD_UNCHANGED)
        self.ebi_img = cv.imread(ebi_img_path, cv.IMREAD_UNCHANGED)
        self.ebi_img = cv.resize(self.ebi_img ,None, fx=1, fy=0.5)
    def put_hand_img(self, extracted_landmark_list, frame):
        for landmark_coordinates in extracted_landmark_list:
            if landmark_coordinates.hand == 'Left':
                img = self.left_hand_img
            elif landmark_coordinates.hand == 'Right':
                img = self.right_hand_img
            x1 = landmark_coordinates.x_min
            x2 = landmark_coordinates.x_max + 1
            y1 = landmark_coordinates.y_min
            y2 = landmark_coordinates.y_max + 1
            img = cv.resize(img, (x2 - x1, y2 - y1))
            frame = self.edit_frame(frame, img, x1, x2, y1, y2)
        return frame
    def put_ebi_img(self, extracted_landmark_list, frame):
        img = self.ebi_img
        x_nine_coordinates = [landmark_coordinates.x_nine for landmark_coordinates in extracted_landmark_list]
        y_nine_coordinates = [landmark_coordinates.y_nine for landmark_coordinates in extracted_landmark_list]
        if x_nine_coordinates:
            x1 = min(x_nine_coordinates)
            x2 = max(x_nine_coordinates) + 1
            y1 = y_nine_coordinates[x_nine_coordinates.index(max(x_nine_coordinates))]
            y2 = y1 + img.shape[0]
            img = cv.resize(img ,(x2 - x1, img.shape[0]))
            frame = self.edit_frame(frame, img, x1, x2, y1, y2)
        return frame
    def edit_frame(self, frame, img, x1, x2, y1, y2):
        if x1 < 0:
            img = img[:, abs(x1):]
            x1 = 0
        if x2 > self.frame_width:
            img = img[:, :-(x2 - self.frame_width)]
            x2 = self.frame_width
        if y1 < 0:
            img = img[abs(y1):, :]
            y1 = 0
        if y2 > self.frame_height:
            img = img[:-(y2 - self.frame_height), :]
            y2 = self.frame_height
        frame[y1:y2, x1:x2] = frame[y1:y2, x1:x2] * (1 - img[:, :, 3:] / 255) + img[:, :, :3] * (img[:, :, 3:] / 255)
        return frame

class ComboMusicController:
    def __init__(self, music_path="path_to_music_file", increase_vol_step=0.05, decrease_vol_step = 0.1):
        self.vol = 0
        self.increase_vol_step = increase_vol_step
        self.decrease_vol_step = decrease_vol_step
        pygame.mixer.init()
        pygame.mixer.music.load(music_path)
        pygame.mixer.music.set_volume(self.vol)
    def increase_vol(self):
        if self.is_busy():
            self.vol += self.increase_vol_step
            if self.vol > 0.5:
                self.vol = 0.5
            pygame.mixer.music.set_volume(self.vol)
        else:
            pygame.mixer.music.play()
    def decrease_vol(self):
        self.vol -= self.decrease_vol_step
        if self.vol <= 0:
            self.vol = 0
            pygame.mixer.music.stop()
        pygame.mixer.music.set_volume(self.vol)
    def is_busy(self):
        return pygame.mixer.music.get_busy()
    
class TouchVisualizer:
    def __init__(self, touch_margin=20):
        self.font_face = cv.FONT_HERSHEY_DUPLEX
        self.margin = 10
        self.font_scale = 1
        self.font_thickness = 1
        self.vibrant_green = (88, 205, 54)
        self.blue = (255, 0, 0)
        size, base_line = cv.getTextSize("A", self.font_face, self.font_scale, self.font_thickness)
        self.font_height = size[1]
        self.line_type = cv.LINE_AA
        self.touch_margin = touch_margin
    def show_touch_text(self, frame, touched_side):
        if touched_side == 'top':
            cv.putText(frame, f"{touched_side}OK!", (self.margin, self.margin + self.font_height), self.font_face, self.font_scale, self.vibrant_green, self.font_thickness, self.line_type)
        elif touched_side == 'bottom':
            cv.putText(frame, f"{touched_side}OK!", (self.margin, (self.margin + self.font_height) * 2), self.font_face, self.font_scale, self.vibrant_green, self.font_thickness, self.line_type)
    def show_combo_num(self, frame, combo_num):
        cv.putText(frame, f"combo:{combo_num}", (self.margin, (self.margin + self.font_height) * 3), self.font_face, self.font_scale, self.vibrant_green, self.font_thickness, self.line_type)
    def draw_touched_marker(self, frame, width, height):
        cv.circle(frame, (width - width // 3, height // 2), self.touch_margin, self.blue)
        cv.circle(frame, (width - width // 3, height - height // 3), self.touch_margin, self.blue)
        return frame
    def draw_hand_marker(self, frame, extracted_landmark_list):
        for landmark_coordinates in extracted_landmark_list:
            cv.drawMarker(frame, (landmark_coordinates.x_nine, landmark_coordinates.y_nine), self.vibrant_green)
        return frame

def main():
    with CvVideoCap() as cap, MpHandLandmarker() as landmarker:
        frame_width = int(cap.width)
        frame_height = int(cap.height)
        ebi_momi_combo_tracker = EbiMomiComboTracker(frame_width, frame_height)
        image_rendere = ImageRendere(frame_width, frame_height)
        combo_music_controller = ComboMusicController()
        touch_visualizer = TouchVisualizer()
        while True:
            frame = cap.read_frame()
            landmarker.detect(frame)
            if landmarker.result is not None:
                landmark_result_extractor = LandmarkResultExtractor(frame_width, frame_height, landmarker.result)
                extracted_landmark_list = landmark_result_extractor.get_coordinates()
                if landmark_result_extractor.is_both_hands():
                    if ebi_momi_combo_tracker.is_combo():
                        frame = image_rendere.put_ebi_img(extracted_landmark_list, frame)
                    touched_side = ebi_momi_combo_tracker.get_touching_side(extracted_landmark_list)
                    if touched_side is not None:
                        touch_visualizer.show_touch_text(frame, touched_side)
                        if ebi_momi_combo_tracker.try_count_combo(touched_side, landmarker.timestamp):
                            if ebi_momi_combo_tracker.is_combo():
                                combo_music_controller.increase_vol()
                if not ebi_momi_combo_tracker.is_moving(landmarker.timestamp):
                    combo_music_controller.decrease_vol()
                    if not combo_music_controller.is_busy():
                        ebi_momi_combo_tracker.reset_touch_state()
                frame = image_rendere.put_hand_img(extracted_landmark_list, frame)

                touch_visualizer.draw_hand_marker(frame, extracted_landmark_list)
            touch_visualizer.show_combo_num(frame, ebi_momi_combo_tracker.combo_num)
            touch_visualizer.draw_touched_marker(frame, frame_width, frame_height)
            cap.show_frame(frame)
            if not cap.wait():
                break
if __name__ == "__main__":
   main()