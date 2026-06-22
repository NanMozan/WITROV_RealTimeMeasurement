import cv2 as cv
import numpy as np

def measureLive():
    # =====================================================================
    # WITROV Real-Time Measurement System
    # Uses two red laser dots as a fixed reference (known distance apart)
    # to calculate real-world dimensions of white objects in frame
    # =====================================================================

    # --- CAMERA SETUP ---
    # Change the first number to select camera:
    # 0 = built-in webcam (default for most laptops)
    # 1 = external/secondary camera (use this for ROV camera)
    cap = cv.VideoCapture(1, cv.CAP_DSHOW)

    if not cap.isOpened():
        print("Cannot open camera — try changing 0 to 1 or vice versa")
        return

    # Kernel for morphological cleanup — 5x5 pixel neighborhood
    # Increase size (e.g. 7,7) to remove larger noise patches
    kernel = np.ones((5, 5), np.uint8)

    while True:
        # Read one frame from camera
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Convert BGR frame to HSV color space
        # HSV separates color (hue) from brightness (value)
        # making color detection more reliable under changing light
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

        # =====================================================================
        # STEP 1 — DETECT RED LASER DOTS
        # Red wraps around in HSV (exists at H=0-10 AND H=170-180)
        # so we need two ranges and combine them
        # =====================================================================

        # Lower red range (H: 0-10)
        redLower1 = np.array([0,   150, 100])
        redUpper1 = np.array([10,  255, 255])

        # Upper red range (H: 170-180)
        # If laser dots are not detected, try lowering 150 to 120
        redLower2 = np.array([170, 150, 100])
        redUpper2 = np.array([180, 255, 255])

        # Create binary masks for each red range
        # inRange returns 255 (white) where color matches, 0 (black) elsewhere
        mask1 = cv.inRange(hsv, redLower1, redUpper1)
        mask2 = cv.inRange(hsv, redLower2, redUpper2)

        # Combine both masks — pixel is white if it matches either red range
        redMask = cv.bitwise_or(mask1, mask2)

        # Find outlines (contours) of white blobs in red mask
        redContours, _ = cv.findContours(redMask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        dotCenters = []  # will store (x, y) center of each detected dot

        for cnt in redContours:
            area = cv.contourArea(cnt)

            # Ignore tiny specks — real laser dots are larger than 50 pixels
            # Increase this value if false detections appear
            if area > 50:

                # Circularity check — laser dots are circles
                # circularity = 1.0 means perfect circle
                # circularity < 0.7 means too irregular to be a laser dot
                perimeter = cv.arcLength(cnt, True)
                if perimeter == 0:
                    continue
                circularity = 4 * np.pi * area / (perimeter ** 2)

                if circularity > 0.7:
                    # Find center of dot using image moments
                    # cx = weighted average x position
                    # cy = weighted average y position
                    M = cv.moments(cnt)
                    if M['m00'] != 0:
                        cx = int(M['m10'] / M['m00'])
                        cy = int(M['m01'] / M['m00'])
                        dotCenters.append((cx, cy))

                        # Draw green circle on frame to show detected dot center
                        cv.circle(frame, (cx, cy), 5, (0, 255, 0), -1)

        # =====================================================================
        # STEP 2 — CALCULATE PIXELS PER CM (reference ratio)
        # Only proceeds if exactly 2 laser dots are found
        # =====================================================================

        if len(dotCenters) == 2:
            x1, y1 = dotCenters[0]
            x2, y2 = dotCenters[1]

            # Euclidean distance between two dot centers in pixels
            # Formula: sqrt((x2-x1)² + (y2-y1)²)
            pixelDistance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

            # IMPORTANT — change this value to match your actual laser dot
            # separation distance in centimeters
            # Measure center-to-center distance of your two lasers
            laserDistanceCm = 9.1  # ← CHANGE THIS to your laser distance in cm

            # Pixels per cm ratio — your live ruler
            # Every measurement divides by this value to convert pixels → cm
            pixelPerCm = pixelDistance / laserDistanceCm

            # Display ratio on frame
            cv.putText(frame, f"px/cm: {pixelPerCm:.1f}",
                      (20, 40), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # =====================================================================
            # STEP 3 — DETECT WHITE OBJECT (PVC pipe / iceberg)
            # White = high brightness (V), low saturation (S), any hue (H)
            # =====================================================================

            # White object HSV range
            # Adjust V min (third value, currently 180) if object appears too dark
            # Adjust S max (second value, currently 60) if background bleeds in
            objLower = np.array([0,   0,  180])
            objUpper = np.array([179, 60, 255])
            objMask = cv.inRange(hsv, objLower, objUpper)

            # Morphological cleanup:
            # MORPH_OPEN  — removes small noise spots (bubbles, reflections)
            # MORPH_CLOSE — fills small holes in the detected object
            objMask = cv.morphologyEx(objMask, cv.MORPH_OPEN, kernel)
            objMask = cv.morphologyEx(objMask, cv.MORPH_CLOSE, kernel)

            # Find object contours
            objContours, _ = cv.findContours(objMask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

            # =====================================================================
            # STEP 4 — MEASURE OBJECT AND DISPLAY DIMENSIONS
            # =====================================================================

            for cnt in objContours:
                area = cv.contourArea(cnt)

                # Ignore small detections — increase if background noise appears
                # Decrease if real object is not being detected
                if area > 2000:

                    # minAreaRect — smallest rectangle fitting around the contour
                    # Works even if object is rotated/tilted
                    rect = cv.minAreaRect(cnt)
                    (cx, cy), (w, h), angle = rect
                    # cx, cy = center of object
                    # w, h   = width and height in pixels
                    # angle  = rotation angle of rectangle

                    # Convert pixel dimensions to real-world cm
                    widthCm  = w / pixelPerCm
                    heightCm = h / pixelPerCm

                    # Get 4 corner points of bounding box and draw it
                    box = cv.boxPoints(rect)
                    box = np.int32(box)
                    cv.drawContours(frame, [box], 0, (255, 0, 0), 2)

                    # Display dimensions on frame next to object
                    cv.putText(frame, f"W: {widthCm:.1f}cm",
                              (int(cx)-40, int(cy)-10),
                              cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                    cv.putText(frame, f"H: {heightCm:.1f}cm",
                              (int(cx)-40, int(cy)+20),
                              cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        # Display the annotated frame
        cv.imshow('WITROV Measurement', frame)

        # Press 'q' to quit cleanly
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    # Release camera and close all windows
    cap.release()
    cv.destroyAllWindows()

if __name__ == '__main__':
    measureLive()