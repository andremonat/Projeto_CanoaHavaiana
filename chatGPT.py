import cv2
import mediapipe as mp

# Inicializa o módulo de detecção de poses do MediaPipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, smooth_landmarks=True, min_detection_confidence=0.5)

# Inicializa uma ferramenta para desenhar os landmarks das poses
mp_drawing = mp.solutions.drawing_utils

# Função para processar cada frame do vídeo
def process_frame(image):
    # Converte a imagem de BGR para RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Processa a imagem para encontrar as poses
    results = pose.process(image_rgb)
    # Desenha os landmarks da pose na imagem
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    return image

# Captura vídeo de um arquivo
cap = cv2.VideoCapture('./videos/demo.mp4')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    # Processa o frame
    processed_frame = process_frame(frame)
    # Exibe o frame processado
    cv2.imshow('Pose Detection', processed_frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
