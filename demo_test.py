#!/usr/bin/env python3

from BlazeposeRenderer import BlazeposeRenderer
import argparse
import streamlit as st
import numpy as np
import mediapipe as mp
import cv2

mp_pose = mp.solutions.pose

parser = argparse.ArgumentParser()
parser.add_argument('-e', '--edge', action="store_true",
                    help="Use Edge mode (postprocessing runs on the device)")
parser_tracker = parser.add_argument_group("Tracker arguments")                 
parser_tracker.add_argument('-i', '--input', type=str, default="rgb", 
                    help="'rgb' or 'rgb_laconic' or path to video/image file to use as input (default=%(default)s)")
parser_tracker.add_argument("--pd_m", type=str,
                    help="Path to an .blob file for pose detection model")
parser_tracker.add_argument("--lm_m", type=str,
                    help="Landmark model ('full' or 'lite' or 'heavy') or path to an .blob file")
parser_tracker.add_argument('-xyz', '--xyz', action="store_true", 
                    help="Get (x,y,z) coords of reference body keypoint in camera coord system (only for compatible devices)")
parser_tracker.add_argument('-c', '--crop', action="store_true", 
                    help="Center crop frames to a square shape before feeding pose detection model")
parser_tracker.add_argument('--no_smoothing', action="store_true", 
                    help="Disable smoothing filter")
parser_tracker.add_argument('-f', '--internal_fps', type=int, 
                    help="Fps of internal color camera. Too high value lower NN fps (default= depends on the model)")                    
parser_tracker.add_argument('--internal_frame_height', type=int, default=640,                                                                                    
                    help="Internal color camera frame height in pixels (default=%(default)i)")                    
parser_tracker.add_argument('-s', '--stats', action="store_true", 
                    help="Print some statistics at exit")
parser_tracker.add_argument('-t', '--trace', action="store_true", 
                    help="Print some debug messages")
parser_tracker.add_argument('--force_detection', action="store_true", 
                    help="Force person detection on every frame (never use landmarks from previous frame to determine ROI)")

parser_renderer = parser.add_argument_group("Renderer arguments")
parser_renderer.add_argument('-3', '--show_3d', choices=[None, "image", "world", "mixed"], default=None,
                    help="Display skeleton in 3d in a separate window. See README for description.")
parser_renderer.add_argument("-o","--output",
                    help="Path to output video file")
 

args = parser.parse_args()

# if args.edge:
#     from BlazeposeDepthaiEdge import BlazeposeDepthai
# else:
#     from BlazeposeDepthai import BlazeposeDepthai
# tracker = BlazeposeDepthai(input_src=args.input, 
#             pd_model=args.pd_m,
#             lm_model=args.lm_m,
#             smoothing=not args.no_smoothing,   
#             xyz=args.xyz,            
#             crop=args.crop,
#             internal_fps=args.internal_fps,
#             internal_frame_height=args.internal_frame_height,
#             force_detection=args.force_detection,
#             stats=True,
#             trace=args.trace)   

# renderer = BlazeposeRenderer(
#                 tracker, 
#                 show_3d=args.show_3d, 
#                 output=args.output)

# while True:
#     # Run blazepose on next frame
#     frame, body = tracker.next_frame()
#     if frame is None: break
#     # Draw 2d skeleton
#     frame = renderer.draw(frame, body)
#     key = renderer.waitKey(delay=1)
#     if key == 27 or key == ord('q'):
#         break
# renderer.exit()
# tracker.exit()
def calculate_angle(a,b,c):
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle >180.0:
        angle = 360-angle
        
    return angle 


def streamlit_cam(args):
    frame_placeholder = st.empty()
    stop_button_pressed = st.button("Stop")

    # if args.edge:
    from BlazeposeDepthaiEdge import BlazeposeDepthai
    # else:
    #     from BlazeposeDepthai import BlazeposeDepthai
    tracker = BlazeposeDepthai(input_src=args.input, 
                pd_model=args.pd_m,
                lm_model=args.lm_m,
                smoothing=not args.no_smoothing,   
                xyz=args.xyz,            
                crop=args.crop,
                internal_fps=args.internal_fps,
                internal_frame_height=args.internal_frame_height,
                force_detection=args.force_detection,
                stats=True,
                trace=args.trace)   

    renderer = BlazeposeRenderer(
                    tracker, 
                    show_3d=args.show_3d, 
                    output=args.output)

    while True:
        # Run blazepose on next frame
        frame, body = tracker.next_frame()
        if frame is None: break
        # Draw 2d skeleton
        frame = renderer.draw(frame, body)
        if body:
            shoulder = [body.landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,body.landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            elbow = [body.landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,body.landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            wrist = [body.landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,body.landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            
            # Calculate angle
            angle = (shoulder, elbow, wrist)
            
            # Visualize angle
            cv2.putText(frame, str(angle), 
                           tuple(np.multiply(elbow, [640, 480]).astype(int)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
            #  Display on 
            frame_placeholder.image(frame,channels="BGR")

        key = renderer.waitKey(delay=1)
        if key == 27 or key == ord('q') or stop_button_pressed:
            break
    renderer.exit()
    tracker.exit()

main_app_sidebar = st.sidebar.title('Inclusive Demo')

#dropdown to select app type
app_type = st.sidebar.selectbox('Which App will you Run Today',('LLM ChatBot', 'Luxonis Camera Demo', 'Other Apps'))

if app_type == 'Luxonis Camera Demo':
        st.subheader(app_type)
        st.title("Webcam Display")
        st.caption("Powered by OpenCV, Streamlit")
        streamlit_cam(args)
