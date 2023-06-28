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
import tkinter as tk
from fer import FER

# Crea una instancia del detector de emociones FER
emo_detector = FER(mtcnn=True)

# Variables para controlar la pausa
pause_time = 5  # Pausa de 5 segundos
pause_start = None

# Emoción detectada previa
previous_emotion = None

# Diccionario de emociones y rutas de videos correspondientes
emotion_videos = {
    'angry': 'angry.mp4',
    'happy': 'happy.mp4',
    'sad': 'sad.mp4',
    'surprise': 'surprise.mp4',
    # Agregar las demás emociones y rutas de videos correspondientes
}

# Función para procesar cada fotograma del video
def process_frame(frame):
    global pause_start, previous_emotion
    
    # Comprueba si se está en pausa
    if pause_start is not None and (cv2.getTickCount() - pause_start) / cv2.getTickFrequency() < pause_time:
        # Muestra el fotograma sin realizar detección de emociones
        cv2.imshow('Emotion Detection', frame)
        return
    
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
        
        # Comprueba si se detecta una emoción dominante distinta a la previa
        if dominant_emotion != previous_emotion:
            previous_emotion = dominant_emotion
            # Muestra la posición y la emoción dominante en el recuadro de la cara
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            text = f"{dominant_emotion}: {emotion_score:.2f}"
            cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            # Comprueba si se detecta una expresión nueva que no sea "neutral"
            if dominant_emotion != 'neutral':
                # Pausa el reconocimiento y muestra la advertencia en pantalla
                pause_start = cv2.getTickCount()
                show_warning(dominant_emotion)

    # Muestra el fotograma procesado en una ventana
    cv2.imshow('Emotion Detection', frame)

# Función para mostrar la advertencia en pantalla
def show_warning(emotion):
    # Obtener la ruta del video correspondiente a la emoción
    video_path = emotion_videos.get(emotion, 'default.mp4')
    
    # Crear una ventana emergente utilizando tkinter
    window = tk.Tk()
    window.title("Nueva Emoción Detectada")
    
    # Crear un widget de reproducción de video
    video_player = tk.Label(window)
    video_player.pack()
    
    # Reproducir el video utilizando OpenCV
    video_capture = cv2.VideoCapture(video_path)
    
    # Función para actualizar el fotograma del video en el widget
    def update_frame():
        ret, frame = video_capture.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame)
            photo = ImageTk.PhotoImage(image)
            video_player.configure(image=photo)
            video_player.image = photo
            video_player.after(30, update_frame)
        else:
            video_capture.release()
            window.destroy()
    
    # Iniciar la reproducción del video
    update_frame()
    
    # Función para cerrar la ventana emergente y detener la reproducción del video
    def close_window():
        video_capture.release()
        window.destroy()
    
    # Cierra la ventana después de 5 segundos
    window.after(int(pause_time * 1000), close_window)
    window.mainloop()

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
