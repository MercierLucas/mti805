import cv2

from utils import FPS


def display_webcam(transform=None):
    cam = cv2.VideoCapture(0)
    cv2.namedWindow('Webcam reader')

    fps_counter = FPS()

    while True:
        ret, frame = cam.read()
        if not ret:
            print('error')
            break

        fps_counter.tick()

        if transform is not None:
            frame = transform(frame)

        cv2.putText(frame, fps_counter.get_fps(), (7, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (100, 255, 0), 3, cv2.LINE_AA)

        cv2.imshow('Webcam reader', frame)

        k = cv2.waitKey(1)
        if k%256 == 27:
            break

    cam.release()
    cv2.destroyAllWindows()