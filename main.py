import asyncio
import sys
import warnings
import os
import pickle
import flet as ft
import base64
import cv2
from keras_facenet import FaceNet
from numpy import expand_dims
import numpy as np
import threading
import time

# Ignorar advertencias de deprecación
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Inicialización de modelos y variables
HaarCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
MyFaceNet = FaceNet()
database = {}
current_signature = None
db_path = './data.pkl'
video_stop_event = threading.Event()  # Evento para controlar la detención del video

# Crear archivo de base de datos si no existe
if not os.path.exists(db_path):
    print("El archivo no existe. Creándolo...")
    with open(db_path, 'wb') as f:
        pickle.dump({}, f)

# Cargar base de datos
def load_database(filename):
    global database
    try:
        with open(filename, "rb") as f:
            database = pickle.load(f)
    except FileNotFoundError:
        print(f"Archivo no encontrado: {filename}. Creando una base de datos vacía.")
        database = {}

# Guardar base de datos
def save_database(filename):
    with open(filename, "wb") as f:
        pickle.dump(database, f)

# Registrar usuario desconocido
def register(page, user_name_field):
    global current_signature
    try:
        user_name = user_name_field.value.strip()
        if not user_name:
            print("Error: El nombre de usuario está vacío.")
            return
        if current_signature is None:
            print("Error: No se detectó ninguna firma facial.")
            return
        
        database[user_name] = current_signature
        save_database(db_path)
        print(f"Firma registrada exitosamente para: {user_name}")
        user_name_field.value = ""
        user_name_field.disabled = True
        page.update()  # Actualiza la interfaz de usuario después del registro
    except Exception as e:
        print(f"Error en register(): {e}")

# Función de captura de video
def capture_video(page: ft.Page, video_image: ft.Image, user_name_field: ft.TextField, register_button: ft.ElevatedButton):
    global current_signature
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: No se pudo abrir la cámara.")
        return

    try:
        while not video_stop_event.is_set():
            success, frame = cap.read()
            if not success:
                print("Error: Fallo al capturar el frame.")
                continue

            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = HaarCascade.detectMultiScale(gray_frame, 1.3, 5)
            person_detected = False
            for (x, y, w, h) in faces:
                person_detected = True
                face = frame[y:y + h, x:x + w]
                face_resized = cv2.resize(face, (160, 160))
                face_expanded = expand_dims(face_resized, axis=0)
                current_signature = MyFaceNet.embeddings(face_expanded)[0]

                identity = 'Desconocido'
                min_dist = 1.2
                for key, value in database.items():
                    dist = np.linalg.norm(value - current_signature)
                    if dist < min_dist:
                        min_dist = dist
                        identity = key

                cv2.putText(frame, identity, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                user_name_field.disabled = identity != "Desconocido"
                register_button.disabled = identity != "Desconocido"
                page.update()

            if not person_detected:
                user_name_field.disabled = True
                register_button.disabled = True
                page.update()

            ret, buffer = cv2.imencode('.jpg', frame)
            if ret:
                frame_bytes = buffer.tobytes()
                base64_img = base64.b64encode(frame_bytes).decode()
                video_image.src_base64 = base64_img
                page.update()

    except Exception as e:
        print(f"Error en video capture: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Video capture thread exited.")

def exit_app(page: ft.Page):
    # Detenemos cualquier hilo o servidor que Flet pueda haber iniciado
    try:
        page.app.quit()  # Debería detener el servidor de Flet
        print("Aplicación de Flet cerrada correctamente.")
    except Exception as e:
        print(f"Error al cerrar la aplicación de Flet: {e}")

    # Intentamos forzar la salida del programa
    sys.exit(0)  # Forzar la terminación completa del proceso

# Función principal de la app
def main(page: ft.Page):
    global video_stop_event, video_thread
    video_stop_event.clear()  # Asegurar que el evento de parada esté limpio al iniciar

    page.bgcolor = ft.colors.BLUE_GREY_800
    page.title = "Reconocimiento Facial"
    page.horizontal_align = ft.CrossAxisAlignment.CENTER
    page.vertical_align = ft.MainAxisAlignment.CENTER  # Centrar también verticalmente
    page.scroll = "adaptive"  # Permitir scroll si el contenido es más grande que la pantalla

    # Texto del título
    t = ft.Text("Video en Tiempo Real", size=20, font_family="RobotoSlab", weight=ft.FontWeight.W_100)

    # Generar imagen base64 inicial como marcador de posición
    placeholder_img = np.zeros((320, 240, 3), dtype=np.uint8)  # Imagen más pequeña
    _, buffer = cv2.imencode('.jpg', placeholder_img)
    placeholder_base64 = base64.b64encode(buffer.tobytes()).decode()

    # Elementos de la interfaz
    video_image = ft.Image(src_base64=placeholder_base64, width=320, height=240)
    user_name_field = ft.TextField(label="Nombre del usuario", width=300, disabled=True)  # Ajustar ancho
    register_button = ft.ElevatedButton('Registrar Rostro', on_click=lambda e: register(page, user_name_field), disabled=True)

    # Botón para salir
    exit_button = ft.ElevatedButton('Salir', on_click=lambda e: exit_app(page))

    # Contenedor compacto
    layout = ft.Column(
        [t, video_image, user_name_field, register_button, exit_button],
        spacing=10,
        horizontal_alignment=ft.CrossAxisAlignment.CENTER  # Alinear el contenido horizontalmente
    )

    # Contenedor externo para centrar todo
    centered_container = ft.Container(
        content=layout,
        alignment=ft.alignment.center,  # Alinear todo al centro
        expand=True  # Usar todo el espacio disponible
    )

    # Agregar diseño al `page`
    page.add(centered_container)

    # Cargar base de datos y lanzar hilo
    load_database(db_path)
    video_thread = threading.Thread(target=capture_video, args=(page, video_image, user_name_field, register_button), daemon=True)
    video_thread.start()

# Iniciar la app
ft.app(target=main)
