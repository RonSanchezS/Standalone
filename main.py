# import cv2
#
# face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#
# cap = cv2.VideoCapture(0)
#
# largest_face = None
#
# while True:
#     _, img = cap.read()
import cv2
from fer import FER

# Crea una instancia del detector de emociones FER
emo_detector = FER(mtcnn=True)

# Función para procesar cada fotograma del video
def process_frame(frame):
    # Detecta las emociones en el fotograma actual
    captured_emotions = emo_detector.detect_emotions(frame)
    # Itera sobre las emociones capturadas en el fotograma
    for face in captured_emotions:
        # Obtiene la posición y las emociones detectadas en la cara
        x, y, w, h = face['box']
        emotions = face['emotions']
        # Encuentra la emoción dominante y su puntuación máxima
        dominant_emotion = max(emotions, key=emotions.get)
        emotion_score = emotions[dominant_emotion]
        # Muestra la posición y la emoción dominante en el recuadro de la cara
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        text = f"{dominant_emotion}: {emotion_score:.2f}"
        cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    # Muestra el fotograma procesado en una ventana
    cv2.imshow('Emotion Detection', frame)

# Inicializa la captura de video desde la cámara
video_capture = cv2.VideoCapture(0)

# Itera sobre los fotogramas capturados de la cámara
while True:
    ret, frame = video_capture.read()
    if not ret:
        break
    # Llama a la función de procesamiento para cada fotograma
    process_frame(frame)
    # Si se presiona 'q', se detiene el bucle
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libera los recursos
video_capture.release()
cv2.destroyAllWindows()


#     height, width, _ = img.shape
#     part_width = int(width / 4)
#
#     img_part = img[:, part_width:part_width * 3, :]
#
#     gray = cv2.cvtColor(img_part, cv2.COLOR_BGR2GRAY)
#
#     faces = face_cascade.detectMultiScale(gray, 1.1, 4)
#
#     largest_area = 0
#     for (x, y, w, h) in faces:
#         x += part_width  # Ajustar las coordenadas con respecto a la imagen original
#         area = w * h
#         if area > largest_area:
#             largest_area = area
#             largest_face = (x, y, w, h)
#
#     if largest_face is not None:
#         x, y, w, h = largest_face
#         cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
#
#     cv2.imshow('img', img)
#
#     k = cv2.waitKey(30) & 0xff
#     if k == 27:
#         break
#
# cap.release()
# cv2.destroyAllWindows()
