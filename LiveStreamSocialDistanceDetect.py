import cv2
import numpy as np
import streamlit as st
import av

from streamlit_webrtc import webrtc_streamer, RTCConfiguration

st.title("Social Distance Maintenance⚠️")
SHOW_VIDEO = True  # Display the video as it is being processed.
INPUT_PATH = 0  # Access my web camara
# Load in the pre-trained SSD model.
configFile = "MobileNetSSD_deploy.prototxt"
modelFile = "MobileNetSSD_deploy.caffemodel"


net = cv2.dnn.readNetFromCaffe(configFile, modelFile)


def detect(frame, network):
    """Detects whether a given frame of contains (un)safe social distancing."""
    results = []
    h, w = frame.shape[:2]

    # Pre-processing: mean subtraction and scaling to match model's training set.
    blob = cv2.dnn.blobFromImage(
        frame, 0.007843, (300, 300), [127.5, 127.5, 127.5])
    network.setInput(blob)

    # Run an inference of the model, passing blob through the network.
    network_output = network.forward()

    # Loop over all results.
    for i in np.arange(0, network_output.shape[2]):
        class_id = network_output[0, 0, i, 1]
        confidence = network_output[0, 0, i, 2]
        if confidence > 0.7 and class_id == 15: # 15 number mean PERSON label have this position
            # Remap 0-1 position outputs to size of image for bounding box.
            box = network_output[0, 0, i, 3:7] * np.array([w, h, w, h])
            box = box.astype('int')

            # Calculate the person center from the bounding box.
            center_x = int((box[0] + box[2]) / 2)
            center_y = int((box[1] + box[3]) / 2)

            results.append((confidence, box, (center_x, center_y)))
    return results


def detect_violations(results):
    """Detects if there are any people who are unsafely close together."""
    violations = set()
    # Multiplier on the pixel width of the smallest detection.
    fac = 1.2

    if len(results) >= 2:
        # Width is right edge minus left.
        boxes_width = np.array([abs(int(r[1][2] - r[1][0])) for r in results])
        centroids = np.array([r[2] for r in results])
        distance_matrix = euclidean_dist(centroids, centroids)
        print("box widths = {}".format(boxes_width))
        print("distancemat = \n{}".format(distance_matrix))

        # For each starting detection...
        for row in range(distance_matrix.shape[0]):
            # Compare distance with every other remaining detection.
            for col in range(row + 1, distance_matrix.shape[1]):
                # Presume unsafe if closer than 1.2x (fac) width of a person apart.
                ref_distance = int(fac * min(boxes_width[row], boxes_width[col]))

                if distance_matrix[row, col] < ref_distance:
                    violations.add(row)
                    violations.add(col)
                print(row, col, violations)
    return violations


def euclidean_dist(A, B):
    """Calculates pair-wise distance between each centroid combination.

    Returns a matrix of len(A) by len(B)."""
    p1 = np.sum(A ** 2, axis=1)[:, np.newaxis]
    p2 = np.sum(B ** 2, axis=1)
    p3 = -2 * np.dot(A, B.T)
    return np.round(np.sqrt(p1 + p2 + p3), 2)


# every time image provide video frame
def callback(img):
    frame = img.to_ndarray(format="bgr24")

    # Detect Boxes.
    results = detect(frame, network=net)

    # Detect boxes too close (i.e. the violations).
    violations = detect_violations(results)

    t, _ = net.getPerfProfile()
    label = 'Inference time: %.2f ms' % (t * 1000.0 / cv2.getTickFrequency())

    # Plot all bounding boxes and whether they are in violation
    for index, (prob, bounding_box, centroid) in enumerate(results):
        start_x, start_y, end_x, end_y = bounding_box

        # Color red if violation, otherwise color green.
        color = (0, 0, 255) if index in violations else (0, 255, 0)
        cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), color, 2)

        cv2.putText(
            frame, label,
            (2, frame.shape[0] - 4),
            cv2.FONT_HERSHEY_TRIPLEX, 0.4, (255, 255, 255))
        cv2.putText(
            frame, 'Not Safe' if index in violations else 'Safe',
            (start_x, start_y - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.putText(
            frame, f'Num Violations: {len(violations)}',
            (10, frame.shape[0] - 25),
            fontFace=cv2.FONT_HERSHEY_PLAIN,
            fontScale=1.0, color=(0, 0, 255), thickness=1)

    return av.VideoFrame.from_ndarray(frame, format="bgr24")



webrtc_streamer(key="Real Time", video_frame_callback=callback, media_stream_constraints={
    "video": True,
    "audio": False},
                rtc_configuration=RTCConfiguration(
                    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
                ))


image_path = "chi.jpg"

st.header("FOUNDER OF CHI👨🏻‍💻")
founder_img = cv2.imread(image_path)
st.image(founder_img[:,:,::-1],width=350)
st.markdown("""We are two brothers **[ ZEN || CHI ]** . Very passionate about learning and building Artificial 
              Intelligence models. Same as you like to eat your favorite food.**We believe Artificial Intelligence 
              solved human any problem in the 21st century.**""")