import cv2
import mediapipe as mp
import numpy as np
import time

# Initialize mediapipe face mesh module
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Drawing specifications for face mesh points
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

# Setup video capture from webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image)
    print(results)
    # Draw the face mesh annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    img_h, img_w, img_c = image.shape
    face_3d = []
    face_2d = []

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            for idx, lm in enumerate(face_landmarks.landmark):
                if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
                    if idx == 1:
                        nose_2d = (lm.x * img_w, lm.y * img_h)
                        nose_3d = (lm.x * img_w, lm.y * img_h, lm.z*3000)

                    x,y = int(lm.x * img_w), int(lm.y * img_h)

                    # Get the 2D Coordinates
                    face_2d.append([x,y])
                    face_3d.append([x,y,lm.z])

            face_2d = np.array(face_2d,dtype=np.float64)
            face_3d = np.array(face_3d,dtype=np.float64)

            focal_length = 1 * img_w
            cam_matrix = np.array([[focal_length, 0, img_h/2],
                                  [0, focal_length, img_w/2],
                                   [0, 0, 1]]
                                  )
            # distortion parameters
            dist_matrix = np.zeros((4, 1), dtype=np.float64)

            success,rot_vec ,trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)

            rmat, jac = cv2.Rodrigues(rot_vec)

            angles,mtxR,mtxQ,Qx,Qy,Qz = cv2.RQDecomp3x3(rmat)

            x = angles[0] * 360
            y = angles[1] * 360
            z = angles[2] * 360

            if y < -10 and x < -10:
                text = "Left Temple"
            elif y > 10 and x > 10:
                text = "Right Temple"
            elif -10 < y < 10 and - 10 < x < 10:
                text = "Frontal Hairline"

            # Display the nose direction
            nose_3d_projection, jacobian = cv2.projectPoints(nose_3d, rot_vec, trans_vec, cam_matrix, dist_matrix)

            p1 = (int(nose_2d[0]), int(nose_2d[1]))
            p2 = (int(nose_2d[0] + y * 10), int(nose_2d[1] - x * 10))

            cv2.line(image, p1, p2, (255, 0, 0), 3)

    # Display the resulting frame
    else:
        text = "Crown"

    # Add the text on the image
    cv2.putText(image, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4)

    cv2.imshow('MediaPipe FaceMesh', image)
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
