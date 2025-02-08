import streamlit as st

import mediapipe as mp
import cv2
import time 

import numpy as np
import tempfile
from PIL import Image

#Lets try to integrate streamlit and mediapipe

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

DEMO_VIDEO = "demo.mp4"
OUTM = "output1.mp4"
DEMO_IMAGE = "demo.jpg"

# print(mp.__version__)
st.title('Instruções para  Análise de Atividades de Canoagem')

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


st.sidebar.title('Dashboard de Análise')
st.sidebar.subheader('Parametros')

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
['About App','Run on Video','Run on Image']
)

if app_mode =='About App':

    st.markdown('In this application we are using **MediaPipe** from Google for creating a Face Mesh on a video')  

    st.text('Demo output for Face mesh')
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
          # About Author \n 
             Hey this is ** Pavan Kunchala ** I hope you like the application \n
            I am looking for ** Collabration ** or ** Freelancing ** in the field of ** Deep Learning ** and 
            ** Computer Vision ** \n
            I am also looking for ** Job opportunities ** in the field of** Deep Learning ** and ** Computer Vision** 
            if you are interested in my profile you can check out my resume from 
            [here](https://drive.google.com/file/d/16aKmdHryldvx3OPNwmHhxW-DAoQOypvX/view?usp=sharing)
            If you're interested in collabrating you can mail me at ** pavankunchalapk@gmail.com ** \n
            You can check out my ** Linkedin ** Profile from [here](https://www.linkedin.com/in/pavan-kumar-reddy-kunchala/) \n
            You can check out my ** Github ** Profile from [here](https://github.com/Pavankunchala) \n
            You can also check my technicals blogs in ** Medium ** from [here](https://pavankunchalapk.medium.com/) \n
            If you are feeling generous you can buy me a cup of ** coffee ** from [here](https://www.buymeacoffee.com/pavankunchala)
             
            ''')
elif app_mode =='Run on Video':

    st.subheader('We are applying Face Mesh on a video')
    # st.set_option('deprecation.showfileUploaderEncoding', False)

    # use_webcam = st.sidebar.button('Use Webcam')
    # record = st.sidebar.checkbox('Record Video')
    # if record:
    #     st.checkbox('Recording',value=True)
    #     st.text('Recording')

    drawing_spec = mp_drawing.DrawingSpec(thickness=2, circle_radius=1)
    

    st.sidebar.text('Params for Video')
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
    detection_confidence = st.sidebar.slider('Min Detection Confidence', min_value =0.0,max_value = 1.0,value = 0.5)
    st.sidebar.markdown('---')
    tracking_confidence = st.sidebar.slider('Min Tracking Confidence', min_value = 0.0,max_value = 1.0,value = 0.5)
    st.sidebar.markdown('---')
    max_faces = st.sidebar.number_input('Maximum Number of Faces',value =5,min_value = 1)
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
        st.markdown('* Frame Rate *')
        kpi1_text = st.markdown('0')
    with kp12:
        st.markdown('* Detected Faces *')
        kpi2_text = st.markdown('0')
    with kpi3:
        st.markdown('* Image Width  *')
        kpi3_text = st.markdown('0')

    st.markdown("<hr/>",unsafe_allow_html=True)

    with mp_face_mesh.FaceMesh(
    max_num_faces=max_faces,
    min_detection_confidence=detection_confidence,
    min_tracking_confidence=tracking_confidence) as face_mesh:
        prevTime = 0


        while vid.isOpened():
            i +=1
            ret, frame = vid.read()
            if not ret:
                continue
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(frame)
            frame.flags.writeable = True
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            face_count = 0
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    mp_drawing.draw_landmarks(
                    image = frame,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=drawing_spec,
                    connection_drawing_spec=drawing_spec)
                    face_count += 1
            currTime = time.time()
            fps = 1/(currTime-prevTime)
            prevTime = currTime

            kpi1_text.write(f'<h1 style=text_alig:center;color:red;>{int(fps)}</hd>',unsafe_allow_html=True)
            kpi2_text.write(f'<h1 style=text_alig:center;color:red;>{face_count}</hd>',unsafe_allow_html=True)
            kpi3_text.write(f'<h1 style=text_alig:center;color:red;>{width}</hd>',unsafe_allow_html=True)
            frame = cv2.resize(frame,(0,0),fx = 0.8 , fy = 0.8)

            frame = image_resize(image = frame, width = 640)
            stframe.image(frame,channels = 'BGR',use_container_width=True) 




elif app_mode =='Run on Image':

    drawing_spec = mp_drawing.DrawingSpec(thickness=2, circle_radius=1)
    
    st.subheader('We are applying Face Mesh on an Image')

    st.markdown('** Number of Detected Faces **')
    kpi1_text=st.markdown('0')

    st.sidebar.text('Params for Image')
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
    detection_confidence = st.sidebar.slider('Min Detection Confidence', min_value =0.0,max_value = 1.0,value = 0.5)
    max_faces = st.sidebar.number_input('Maximum Number of Faces',value =2,min_value = 1)
    face_count = 0
    st.sidebar.markdown('---')

    img_file_buffer = st.sidebar.file_uploader("Upload an image", type=[ "jpg", "jpeg",'png'])

    if img_file_buffer is not None:
        image = np.array(Image.open(img_file_buffer))

    else:
        demo_image = DEMO_IMAGE
        image = np.array(Image.open(demo_image))

    st.sidebar.markdown('---')

    st.sidebar.text('Original Image')
    st.sidebar.image(image)


    with mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=max_faces,
    min_detection_confidence=detection_confidence) as face_mesh:
    


        results = face_mesh.process(image)

        out_image = image.copy()

        for face_landmarks in results.multi_face_landmarks:
            #print('face_landmarks:', face_landmarks)
            face_count += 1

            mp_drawing.draw_landmarks(
            image=out_image,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=drawing_spec,
            connection_drawing_spec=drawing_spec)

        kpi1_text.write(f'{face_count}')
        st.sidebar.markdown('---')

        st.subheader('Output Image')

        

        st.image(out_image,use_column_width= True)

        
