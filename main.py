import cv2
import numpy as np
from ultralytics import YOLO


# ============================================================
# 1. LOAD MODEL
# ============================================================

# IMPORTANT:
#
# yolov8n.pt is the normal COCO model.
# It can detect PERSON, but it cannot detect HELMET
# or SAFETY-VEST.
#
# If you have a custom construction safety model, use:
#
# model = YOLO("safety_gear.pt")
#
# The custom model should contain:
#   person
#   helmet
#   safety-vest
#
model = YOLO("safety_gear.pt")


# ============================================================
# 2. VIDEO SETTINGS
# ============================================================

input_video = "videos/input_video.mp4"
output_video = "videos/output_video.mp4"


# Open video
cap = cv2.VideoCapture(input_video)


if not cap.isOpened():
    print("ERROR: Could not open video.")
    exit()


# Get video information
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)


print("Video width :", width)
print("Video height:", height)
print("Video FPS   :", fps)


# ============================================================
# 3. CREATE OUTPUT VIDEO
# ============================================================

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

out = cv2.VideoWriter(
    output_video,
    fourcc,
    fps,
    (width, height)
)


# ============================================================
# 4. DANGER ZONE
# ============================================================

# IMPORTANT:
#
# These coordinates are only an example.
#
# You MUST adjust them according to your camera.
#
# For a 1920x1080 video, for example:
#
danger_zone = np.array([
    [700, 300],
    [1250, 300],
    [1500, 900],
    [450, 900]
], np.int32)


# ============================================================
# 5. PROCESS VIDEO
# ============================================================

while True:

    success, frame = cap.read()

    if not success:
        break


    # ========================================================
    # RUN YOLO
    # ========================================================

    results = model(
        frame,
        conf=0.40,
        verbose=False
    )


    # ========================================================
    # DRAW TRANSPARENT DANGER ZONE
    # ========================================================

    overlay = frame.copy()

    cv2.fillPoly(
        overlay,
        [danger_zone],
        (0, 0, 255)
    )

    # Make the red zone transparent
    frame = cv2.addWeighted(
        overlay,
        0.20,
        frame,
        0.80,
        0
    )


    # Draw only the OUTLINE of the danger zone
    cv2.polylines(
        frame,
        [danger_zone],
        True,
        (0, 0, 255),
        3
    )


    # ========================================================
    # STORE DETECTED OBJECTS
    # ========================================================

    persons = []
    helmets = []
    vests = []


    # ========================================================
    # READ YOLO DETECTIONS
    # ========================================================

    for result in results:

        for box in result.boxes:

            class_id = int(box.cls[0])
            confidence = float(box.conf[0])

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0]
            )

            class_name = model.names[class_id]


            # ------------------------------------------------
            # PERSON
            # ------------------------------------------------

            if class_name == "person":

                persons.append({
                    "box": (x1, y1, x2, y2),
                    "confidence": confidence
                })


            # ------------------------------------------------
            # HELMET
            # ------------------------------------------------

            elif class_name == "helmet":

                helmets.append({
                    "box": (x1, y1, x2, y2),
                    "confidence": confidence
                })


            # ------------------------------------------------
            # SAFETY VEST
            # ------------------------------------------------

            elif class_name in ["safety-vest", "vest"]:

                vests.append({
                    "box": (x1, y1, x2, y2),
                    "confidence": confidence
                })


    # ========================================================
    # CHECK EACH PERSON
    # ========================================================

    danger_violation = False


    for person in persons:

        x1, y1, x2, y2 = person["box"]


        # ====================================================
        # CHECK DANGER ZONE
        # ====================================================

        # Use the person's CENTER point.
        #
        # This prevents random detections around the person
        # from triggering the danger zone.

        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)


        inside_zone = cv2.pointPolygonTest(
            danger_zone,
            (center_x, center_y),
            False
        )


        if inside_zone >= 0:
            danger_violation = True


        # ====================================================
        # CHECK FOR HELMET
        # ====================================================

        wearing_helmet = False


        for helmet in helmets:

            hx1, hy1, hx2, hy2 = helmet["box"]


            # Helmet center
            helmet_center_x = int((hx1 + hx2) / 2)
            helmet_center_y = int((hy1 + hy2) / 2)


            # Check whether helmet is inside the upper
            # portion of the person's bounding box.
            #
            # This helps associate the helmet with the
            # correct person.

            if (
                x1 <= helmet_center_x <= x2
                and
                y1 <= helmet_center_y <= y1 + int((y2 - y1) * 0.45)
            ):

                wearing_helmet = True
                break


        # ====================================================
        # CHECK FOR SAFETY VEST
        # ====================================================

        wearing_vest = False


        for vest in vests:

            vx1, vy1, vx2, vy2 = vest["box"]


            # Vest center
            vest_center_x = int((vx1 + vx2) / 2)
            vest_center_y = int((vy1 + vy2) / 2)


            # Vest should be somewhere around the person's
            # upper/middle body.

            if (
                x1 <= vest_center_x <= x2
                and
                y1 + int((y2 - y1) * 0.20)
                <= vest_center_y
                <=
                y1 + int((y2 - y1) * 0.75)
            ):

                wearing_vest = True
                break


        # ====================================================
        # DETERMINE SAFETY STATUS
        # ====================================================

        if wearing_helmet and wearing_vest:

            # Green = everything is present
            box_color = (0, 255, 0)

            status = "SAFE"

        else:

            # Red = something is missing
            box_color = (0, 0, 255)

            missing = []

            if not wearing_helmet:
                missing.append("HELMET")

            if not wearing_vest:
                missing.append("VEST")

            status = "MISSING: " + ", ".join(missing)


        # ====================================================
        # DRAW PERSON BOX
        # ====================================================

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            box_color,
            3
        )


        # ====================================================
        # DRAW STATUS
        # ====================================================

        cv2.putText(
            frame,
            status,
            (x1, max(y1 - 10, 25)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            box_color,
            2
        )


    # ========================================================
    # DRAW HELMET BOXES
    # ========================================================

    for helmet in helmets:

        x1, y1, x2, y2 = helmet["box"]

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            "HELMET",
            (x1, max(y1 - 5, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2
        )


    # ========================================================
    # DRAW VEST BOXES
    # ========================================================

    for vest in vests:

        x1, y1, x2, y2 = vest["box"]

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            "SAFETY VEST",
            (x1, max(y1 - 5, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2
        )


    # ========================================================
    # DANGER ZONE ALERT
    # ========================================================

    if danger_violation:

        text = "DANGER ZONE VIOLATION!"

        text_size = cv2.getTextSize(
            text,
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            4
        )[0]


        text_width = text_size[0]


        # Center the warning
        text_x = (width - text_width) // 2


        # Black background
        cv2.rectangle(
            frame,
            (text_x - 20, 20),
            (text_x + text_width + 20, 85),
            (0, 0, 0),
            -1
        )


        # Red warning
        cv2.putText(
            frame,
            text,
            (text_x, 68),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 0, 255),
            4
        )


    # ========================================================
    # SHOW VIDEO
    # ========================================================

    cv2.imshow(
        "Construction Worker Safety Monitor",
        frame
    )


    # ========================================================
    # SAVE VIDEO
    # ========================================================

    out.write(frame)


    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# ============================================================
# CLEAN UP
# ============================================================

cap.release()
out.release()
cv2.destroyAllWindows()


print()
print("======================================")
print("Processing finished!")
print("Output:", output_video)
print("======================================")
