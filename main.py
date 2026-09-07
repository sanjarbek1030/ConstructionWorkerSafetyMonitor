"""
=============================================================================
 CONSTRUCTION WORKER SAFETY MONITOR
=============================================================================
 What this script does:
   1. Opens a construction-site video.
   2. Runs a YOLOv8 model on every frame to find people, helmets,
      and safety vests.
   3. Checks whether each detected person appears to have a helmet
      and safety vest.
   4. Workers WITH both helmet and vest get a GREEN box.
   5. Workers MISSING helmet or vest get a RED box.
   6. The detected helmet and safety-vest boxes are also displayed.
   7. Shows the processed video live.
   8. Saves the processed video to:
         videos/output_video.mp4

 REQUIRED PACKAGES:
     pip install ultralytics opencv-python numpy

 IMPORTANT NOTE ABOUT THE MODEL:
   The standard 'yolov8n.pt' model is trained on the COCO dataset.
   COCO contains the "person" class, but it does NOT contain
   "helmet" or "safety-vest".

   Therefore, for real helmet/vest detection, you need a custom
   YOLO model trained on construction safety data.

   For example:
       model = YOLO("safety_gear.pt")

   Your custom model should contain classes similar to:
       person
       helmet
       safety-vest

   The class-matching code below is flexible and can also recognize
   names such as:
       Helmet
       hard-hat
       safety helmet
       vest
       safety vest
=============================================================================
"""

import cv2
import numpy as np
from ultralytics import YOLO


# =============================================================================
# SECTION 1: CONFIGURATION
# =============================================================================

# --- Video files ---

INPUT_VIDEO_PATH = "videos/input_video.mp4"

OUTPUT_VIDEO_PATH = "videos/output_video.mp4"


# --- YOLO model ---

# Use your custom safety-gear model here.
#
# IMPORTANT:
# "yolov8n.pt" can detect people but NOT helmets or safety vests.
#
# If your custom model is called safety_gear.pt:
#
#     MODEL_PATH = "safety_gear.pt"
#
MODEL_PATH = "safety_gear.pt"


# Minimum confidence required for a detection.
#
# 0.4 means that YOLO will ignore detections with less than
# 40% confidence.
#
# You can try:
#
# 0.30 = more detections
# 0.40 = balanced
# 0.50 = stricter
#
CONFIDENCE_THRESHOLD = 0.4


# =============================================================================
# SECTION 2: COLORS
# =============================================================================

# OpenCV uses BGR colors instead of RGB.

COLOR_RED = (0, 0, 255)

COLOR_GREEN = (0, 255, 0)

COLOR_WHITE = (255, 255, 255)


# =============================================================================
# SECTION 3: HELPER FUNCTIONS
# =============================================================================

def boxes_overlap(box_a, box_b):
    """
    Checks whether two rectangular bounding boxes overlap.

    box format:
        (x1, y1, x2, y2)

    Returns:
        True  -> boxes overlap
        False -> boxes do not overlap
    """

    ax1, ay1, ax2, ay2 = box_a

    bx1, by1, bx2, by2 = box_b


    # Check if box A is completely to the left/right of box B.
    if ax2 < bx1 or bx2 < ax1:
        return False


    # Check if box A is completely above/below box B.
    if ay2 < by1 or by2 < ay1:
        return False


    # Otherwise, the boxes overlap.
    return True


def match_class_name(model_class_names, keyword):
    """
    Searches through the model's class names.

    For example, if the model contains:

        0: person
        1: helmet
        2: safety-vest

    Then:

        match_class_name(model.names, "helmet")

    will find:

        ["helmet"]

    Matching is case-insensitive.
    """

    matches = []


    for class_id, class_name in model_class_names.items():

        if keyword.lower() in class_name.lower():

            matches.append(class_name)


    return matches


# =============================================================================
# SECTION 4: LOAD YOLO MODEL
# =============================================================================

print("Loading YOLO model...")

model = YOLO(MODEL_PATH)


# Get the class names from the model.
#
# Example:
#
# {
#     0: "person",
#     1: "helmet",
#     2: "safety-vest"
# }
#
person_class_names = match_class_name(
    model.names,
    "person"
)

helmet_class_names = match_class_name(
    model.names,
    "helmet"
)

vest_class_names = match_class_name(
    model.names,
    "vest"
)


print()
print("Model classes:")
print(model.names)

print()
print("Detected class groups:")
print("Person :", person_class_names)
print("Helmet :", helmet_class_names)
print("Vest   :", vest_class_names)
print()


# Warn the user if the model doesn't contain helmet/vest classes.
if not helmet_class_names or not vest_class_names:

    print("============================================================")
    print("WARNING:")
    print("This model does not contain helmet and/or vest classes.")
    print()
    print("The standard yolov8n.pt model cannot detect helmets")
    print("or safety vests.")
    print()
    print("Use a custom safety-gear trained model such as:")
    print("    safety_gear.pt")
    print("============================================================")
    print()


# =============================================================================
# SECTION 5: OPEN INPUT VIDEO
# =============================================================================

print("Opening video...")

video_capture = cv2.VideoCapture(
    INPUT_VIDEO_PATH
)


# Make sure the video opened correctly.
if not video_capture.isOpened():

    raise IOError(
        f"Could not open video file: {INPUT_VIDEO_PATH}"
    )


# Get video dimensions.
frame_width = int(
    video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)
)

frame_height = int(
    video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
)


# Get video FPS.
frames_per_second = video_capture.get(
    cv2.CAP_PROP_FPS
)


# Sometimes a video doesn't contain valid FPS information.
# Use 30 FPS as a fallback.
if frames_per_second <= 0:

    frames_per_second = 30.0


print(
    f"Input video: "
    f"{frame_width}x{frame_height} "
    f"@ {frames_per_second:.2f} FPS"
)


# =============================================================================
# SECTION 6: CREATE OUTPUT VIDEO
# =============================================================================

# MP4 video codec.
fourcc = cv2.VideoWriter_fourcc(
    *"mp4v"
)


# Create the output video writer.
#
# We use:
# - same width
# - same height
# - same FPS
#
video_writer = cv2.VideoWriter(

    OUTPUT_VIDEO_PATH,

    fourcc,

    frames_per_second,

    (frame_width, frame_height)
)


# =============================================================================
# SECTION 7: MAIN VIDEO LOOP
# =============================================================================

print()
print("Starting video processing...")
print("Press 'q' to stop.")
print()


frame_count = 0


while True:

    # ---------------------------------------------------------
    # Read one frame from the video.
    # ---------------------------------------------------------

    success, frame = video_capture.read()


    # If there are no more frames, stop.
    if not success:

        break


    frame_count += 1


    # ---------------------------------------------------------
    # Run YOLO detection.
    # ---------------------------------------------------------

    results = model(

        frame,

        conf=CONFIDENCE_THRESHOLD,

        verbose=False

    )[0]


    # ---------------------------------------------------------
    # Create lists for our three object types.
    # ---------------------------------------------------------

    person_boxes = []

    helmet_boxes = []

    vest_boxes = []


    # ---------------------------------------------------------
    # Read all YOLO detections.
    # ---------------------------------------------------------

    for detected_box in results.boxes:


        # Get class ID.
        class_id = int(
            detected_box.cls[0]
        )


        # Get class name.
        class_name = model.names[class_id]


        # Get bounding-box coordinates.
        x1, y1, x2, y2 = map(
            int,
            detected_box.xyxy[0]
        )


        # Store the bounding box.
        box_coords = (
            x1,
            y1,
            x2,
            y2
        )


        # -----------------------------------------------------
        # PERSON
        # -----------------------------------------------------

        if class_name in person_class_names:

            person_boxes.append(
                box_coords
            )


        # -----------------------------------------------------
        # HELMET
        # -----------------------------------------------------

        elif class_name in helmet_class_names:

            helmet_boxes.append(
                box_coords
            )


        # -----------------------------------------------------
        # SAFETY VEST
        # -----------------------------------------------------

        elif class_name in vest_class_names:

            vest_boxes.append(
                box_coords
            )


    # =============================================================================
    # SECTION 8: CHECK EVERY PERSON
    # =============================================================================

    for person_box in person_boxes:


        x1, y1, x2, y2 = person_box


        # -----------------------------------------------------
        # CHECK FOR HELMET
        # -----------------------------------------------------

        has_helmet = False


        for helmet_box in helmet_boxes:


            # Check whether the helmet overlaps the person.
            if boxes_overlap(
                person_box,
                helmet_box
            ):

                has_helmet = True

                break


        # -----------------------------------------------------
        # CHECK FOR SAFETY VEST
        # -----------------------------------------------------

        has_vest = False


        for vest_box in vest_boxes:


            # Check whether the vest overlaps the person.
            if boxes_overlap(
                person_box,
                vest_box
            ):

                has_vest = True

                break


        # -----------------------------------------------------
        # DETERMINE SAFETY STATUS
        # -----------------------------------------------------

        if has_helmet and has_vest:

            # Person has both required pieces of PPE.
            box_color = COLOR_GREEN

            status_text = "SAFE"


        else:

            # Something is missing.
            box_color = COLOR_RED


            missing_items = []


            if not has_helmet:

                missing_items.append(
                    "HELMET"
                )


            if not has_vest:

                missing_items.append(
                    "VEST"
                )


            status_text = (
                "MISSING: "
                + ", ".join(missing_items)
            )


        # -----------------------------------------------------
        # DRAW PERSON BOX
        # -----------------------------------------------------

        cv2.rectangle(

            frame,

            (x1, y1),

            (x2, y2),

            box_color,

            3

        )


        # -----------------------------------------------------
        # DRAW PERSON STATUS
        # -----------------------------------------------------

        cv2.putText(

            frame,

            status_text,

            (x1, max(y1 - 10, 20)),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.6,

            box_color,

            2,

            cv2.LINE_AA

        )


    # =============================================================================
    # SECTION 9: DRAW HELMET DETECTIONS
    # =============================================================================

    for helmet_box in helmet_boxes:


        x1, y1, x2, y2 = helmet_box


        cv2.rectangle(

            frame,

            (x1, y1),

            (x2, y2),

            COLOR_GREEN,

            2

        )


        cv2.putText(

            frame,

            "HELMET",

            (x1, max(y1 - 5, 20)),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.5,

            COLOR_GREEN,

            2,

            cv2.LINE_AA

        )


    # =============================================================================
    # SECTION 10: DRAW SAFETY VEST DETECTIONS
    # =============================================================================

    for vest_box in vest_boxes:


        x1, y1, x2, y2 = vest_box


        cv2.rectangle(

            frame,

            (x1, y1),

            (x2, y2),

            COLOR_GREEN,

            2

        )


        cv2.putText(

            frame,

            "SAFETY VEST",

            (x1, max(y1 - 5, 20)),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.5,

            COLOR_GREEN,

            2,

            cv2.LINE_AA

        )


    # =============================================================================
    # SECTION 11: INFORMATION DISPLAY
    # =============================================================================

    info_text = (
        f"Frame: {frame_count} | "
        f"Workers: {len(person_boxes)}"
    )


    cv2.putText(

        frame,

        info_text,

        (10, frame_height - 15),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.5,

        COLOR_WHITE,

        1,

        cv2.LINE_AA

    )


    # =============================================================================
    # SECTION 12: SAVE FRAME
    # =============================================================================

    video_writer.write(
        frame
    )


    # =============================================================================
    # SECTION 13: DISPLAY FRAME
    # =============================================================================

    cv2.imshow(

        "Construction Worker Safety Monitor",

        frame

    )


    # Press Q to stop.
    if cv2.waitKey(1) & 0xFF == ord("q"):

        print("Stopped by user.")

        break


# =============================================================================
# SECTION 14: CLEAN UP
# =============================================================================

video_capture.release()

video_writer.release()

cv2.destroyAllWindows()


print()
print("============================================================")
print("Processing finished!")
print(f"Frames processed: {frame_count}")
print(f"Output video: {OUTPUT_VIDEO_PATH}")
print("============================================================")
