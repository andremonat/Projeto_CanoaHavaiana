import streamlit as st

import mediapipe as mp
import cv2
import time 

import numpy as np
import tempfile
from PIL import Image

# print(st.__version__)
# print(cv2.__version__)
# print(mp.__version__)


#Lets try to integrate streamlit and mediapipe

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils 
mp_drawing_styles = mp.solutions.drawing_styles
#mp_face_mesh = mp.solutions.face_mesh

DEMO_VIDEO = "./videos/Homem_Academia.mp4"
OUTM = "output1.mp4"
DEMO_IMAGE = "./images/remadorSolitario.jpg"

# print(mp.__version__)
st.title('Aplicativo para Análise de Movimentos de Canoagem Havaiana')

st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        width: 350px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 350px;
        margin-left: -350px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


st.sidebar.title('Condições para Análise de Poses')
st.sidebar.subheader('Parâmetros')

@st.cache_resource()
def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)

    # return the resized image
    return resized




app_mode = st.sidebar.selectbox('Choose the App mode',
['Sobre o Aplicativo','Processar Video','Processar Imagem']
)

if app_mode =='Sobre o Aplicativo':

    st.markdown('O Objetivo desse Aplicativo é analisar os movimentos de atletas de canoagem havaiana')  

    st.text('Versão Inicial com fins demonstrativos')
    st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        width: 350px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 350px;
        margin-left: -350px;
    }
    </style>
    """,
    unsafe_allow_html=True,
    )
    dem = open(DEMO_VIDEO,"rb")
    out_vid = dem.read()

    st.video(out_vid)

    st.markdown('''
            #Sobre o Ron \n 
            Olá! Eu sou o Ron \n
            Com muitos anos de Canoagem Havaiana estou realizando meu sonho de criar um assistente inteligente para te auxiliar no nosso esporte! \n
            Adoraria receber seu feedback! \n
            Por favor me escreva ! \n
            email: canoagemhavainaIA@gmail.com \n
            Celular: 21 99999-9999 \n
            Instagram: @canoagemhavainaIA \n
             
            ''')
elif app_mode =='Processar Video':

    st.subheader('Análise de Poses na Canoa em Vídeo')
    # st.set_option('deprecation.showfileUploaderEncoding', False)

    # use_webcam = st.sidebar.button('Use Webcam')
    # record = st.sidebar.checkbox('Record Video')
    # if record:
    #     st.checkbox('Recording',value=True)
    #     st.text('Recording')

    drawing_spec = mp_drawing.DrawingSpec(thickness=4, circle_radius=1)
    

    st.sidebar.text('Parâmetros para Análise do Video')
    st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        width: 350px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 350px;
        margin-left: -350px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
    st.sidebar.markdown('---')
    detection_confidence = st.sidebar.slider('Confiança mínima de Detecção', min_value =0.0,max_value = 1.0,value = 0.5)
    st.sidebar.markdown('---')
    tracking_confidence = st.sidebar.slider('Confiança Mínima de Acompanhamento', min_value = 0.0,max_value = 1.0,value = 0.5)
    st.sidebar.markdown('---')
    max_atletas = st.sidebar.number_input('Número Máximo de Atletas',value =5,min_value = 1)
    st.sidebar.markdown('---')

    stframe = st.empty()

    video_file_buffer = st.sidebar.file_uploader("Upload a video", type=[ "mp4", "mov",'avi','asf', 'm4v' ])
    tffile = tempfile.NamedTemporaryFile(delete=False)

    if not video_file_buffer:
        # if use_webcam:
        #     vid = cv2.VideoCapture(0)
        # else:
        vid = cv2.VideoCapture(DEMO_VIDEO)
        tffile.name = DEMO_VIDEO   
    else:
        tffile.write(video_file_buffer.read())
        vid = cv2.VideoCapture(tffile.name)
    

    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_input = int(vid.get(cv2.CAP_PROP_FPS))
    #codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
    codec = cv2.VideoWriter_fourcc('V','P','0','9')
    out = cv2.VideoWriter('output1.mp4', codec, fps_input, (width, height))
    st.sidebar.text('Input Video')
    st.sidebar.video(tffile.name)

    fps=0
    i=0 

    drawing_spec = mp_drawing.DrawingSpec(thickness=2, circle_radius=2)

    kpi1,kp12,kpi3 = st.columns(3)

    with kpi1:
        st.markdown('* Quadros Analisados por segundo*')
        kpi1_text = st.markdown('0')
    with kp12:
        st.markdown('* Número de Atletas Detectados *')
        kpi2_text = st.markdown('0')
    with kpi3:
        st.markdown('* Largura da Imagem *')
        kpi3_text = st.markdown('0')

    st.markdown("<hr/>",unsafe_allow_html=True)

    # with mp_face_mesh.FaceMesh(
    # max_num_faces=max_atletas,
    # min_detection_confidence=detection_confidence,
    # min_tracking_confidence=tracking_confidence) as face_mesh:
    with mp_pose.Pose(
        static_image_mode=True, min_detection_confidence=detection_confidence, model_complexity=2) as pose:
         _,frame = vid.read()
         # pose= mp_pose.Pose()
         results = pose.process(frame)
         mp_drawing.draw_landmarks(      
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
         prevTime = 0


         while vid.isOpened():
            i +=1
            ret, frame = vid.read()
            if not ret:
                continue
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frame)
            frame.flags.writeable = True
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            atlete_count = 0
            # print(len(results.pose_landmarks))
            if results.pose_landmarks:          
                for pose_landmark in results.pose_landmarks.landmark:
                    mp_drawing.draw_landmarks(      
                    frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
                atlete_count += 1

            currTime = time.time()
            fps = 1/(currTime-prevTime)
            prevTime = currTime

            kpi1_text.write(f'<h1 style=text_alig:center;color:red;>{int(fps)}</hd>',unsafe_allow_html=True)
            kpi2_text.write(f'<h1 style=text_alig:center;color:red;>{atlete_count}</hd>',unsafe_allow_html=True)
            kpi3_text.write(f'<h1 style=text_alig:center;color:red;>{width}</hd>',unsafe_allow_html=True)
            frame = cv2.resize(frame,(0,0),fx = 0.8 , fy = 0.8)

            frame = image_resize(image = frame, width = 640)
            stframe.image(frame,channels = 'BGR',use_container_width=True) 
           




elif app_mode =='Processar Imagem':

    drawing_spec = mp_drawing.DrawingSpec(thickness=2, circle_radius=1)
    
    st.subheader('nálise de Poses de Atletas na Imagem da Canoa ')
    atlete_count = 0
    quadros = 1
    kpi1,kp12,kpi3 = st.columns(3)

    with kpi1:
        st.markdown('* Quadros Analisados  *')
        kpi1_text = st.markdown('0')
    with kp12:
        st.markdown('* Número de Atletas Detectados *')
        kpi2_text = st.markdown('0')
    with kpi3:
        st.markdown('* Largura da Imagem *')
        kpi3_text = st.markdown('0')

    st.sidebar.text('Parâmetrs da Imagem')
    st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        width: 350px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 350px;
        margin-left: -350px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
    st.sidebar.markdown('---')
    detection_confidence = st.sidebar.slider('Confiança Mínima de Detecção', min_value =0.0,max_value = 1.0,value = 0.5)
    max_faces = st.sidebar.number_input('Número Máximo de Atletas',value =2,min_value = 1)
    atlete_count_count = 0
    st.sidebar.markdown('---')

    img_file_buffer = st.sidebar.file_uploader("Upload da image", type=[ "jpg", "jpeg",'png'])

    if img_file_buffer is not None:
        image = np.array(Image.open(img_file_buffer))

    else:
        demo_image = DEMO_IMAGE
        image = np.array(Image.open(demo_image))
    (h, w) = image.shape[:2]
    st.sidebar.markdown('---')

    st.sidebar.text('Imagem Original')
    st.sidebar.image(image)


    with mp_pose.Pose(
        static_image_mode=True, min_detection_confidence=detection_confidence, model_complexity=2) as pose:
    

        results = pose.process(image)

       

       

        # for face_landmarks in results.multi_face_landmarks:
        #     #print('face_landmarks:', face_landmarks)
        #     face_count += 1

        #     mp_drawing.draw_landmarks(
        #     image=out_image,
        #     landmark_list=face_landmarks,
        #     connections=mp_face_mesh.FACEMESH_CONTOURS,
        #     landmark_drawing_spec=drawing_spec,
        #     connection_drawing_spec=drawing_spec)

        width = w

        if results.pose_landmarks: 
                atlete_count += 1         
                for pose_landmark in results.pose_landmarks.landmark:
                    mp_drawing.draw_landmarks(      
                    image,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
        kpi1_text.write(f'<h1 style=text_alig:center;color:red;>{int(quadros)}</hd>',unsafe_allow_html=True)
        kpi2_text.write(f'<h1 style=text_alig:center;color:red;>{atlete_count}</hd>',unsafe_allow_html=True)
        kpi3_text.write(f'<h1 style=text_alig:center;color:red;>{width}</hd>',unsafe_allow_html=True)

        out_image = image.copy()
        st.sidebar.markdown('---')

        st.subheader('Output Image')

    

        st.image(out_image,use_container_width= True)
        kpi1_text.write(f'<h1 style=text_alig:center;color:red;>{atlete_count}</hd>',unsafe_allow_html=True)

        
