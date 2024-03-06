import cv2
import streamlit as st
import os
from ultralytics import YOLO
import math
import numpy as np
import cvzone
import pandas as pd
import streamlit as st

def streamlitApp():
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    st.title("🎦 AI Construction Site Surveillance")
    # st.markdown("---")
    
    video_folder = 'testing-videos'
    video_files = [f for f in os.listdir(video_folder) if f.endswith(('.mp4', '.avi'))]
    # video_files.insert(0,"Webcam")
    video_files.insert(0, "Select a video file or webcam")
    st.sidebar.title("Video Source")
    selected_video = st.sidebar.selectbox("", video_files, index=0)

    if selected_video == "Webcam":
        cap = cv2.VideoCapture(0)
        # cap.set(3, 960)
        # cap.set(4, 540)
    else:
        cap = cv2.VideoCapture(os.path.join(video_folder, selected_video))
    model = YOLO("models/yolov8n.pt")
    max_id = -1
    counter = 0
    new_labels = ["Helmet", "Vest"]
    d = {}

    if not cap.isOpened():
        st.info("Select a video file or webcam to start the app.")
        return
    with st.container(border=True):
        image_placeholder = st.empty()
    if selected_video != "Select a video file or webcam":
        start_button, stop_button = st.sidebar.columns(2)
        start_button = start_button.button("Start Video", type="primary",use_container_width=True)
        stop_button = stop_button.button("Stop Video", type="primary",use_container_width=True)
    st.sidebar.markdown("---")
    st.sidebar.title("Results")
    result_placeholder = st.sidebar.empty()

    while start_button and not stop_button:
        success, img = cap.read()
        if not success:
            st.info("Video ended Successfully! Select another video file or webcam to continue.")
            break     

        results = model.track(img, persist=True, classes=0)
        
        for r in results:
            boxes = r.boxes
            ids = set()
        
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                w, h = x2 - x1, y2 - y1
            
                if box.id == None:
                    d = {}
                    counter = 0
            
                if box.id != None:
                    ids.add(box.id[0].item())
                
                    if box.id[0].item() not in d:
                        d[box.id[0].item()] = []
                    d[box.id[0].item()].extend([counter, "Not Helmet", "Not Vest", x1, y1, x2, y2])
                    
                    if box.id[0] > max_id:
                        counter += 1
                        n_x1, n_y1, n_x2, n_y2 = box.xyxy[0]
                        n_x1, n_y1, n_x2, n_y2 = int(n_x1), int(n_y1), int(n_x2), int(n_y2)
                        w, h = n_x2 - n_x1, n_y2 - n_y1
                
                        d[box.id[0].item()][0] = counter
                    
                        canvas = np.zeros(img.shape, dtype=np.uint8)
                        cv2.rectangle(canvas, (n_x1, n_y1), (n_x2, n_y2), (255, 255, 255), thickness=cv2.FILLED)
                        person = cv2.bitwise_and(img, canvas)
                
                        new_model = YOLO("models/best.pt")
                        new_results = new_model(person)
            
                        for new_r in new_results:
                            new_boxes = new_r.boxes
                            new_coor = []
        
                            if len(new_boxes) == 0:
                                new_coor.append(["Not Helmet"])
                                new_coor.append(["Not Vest"])
                                continue
        
                            for new_box in new_boxes:
                                new_x1, new_y1, new_x2, new_y2 = new_box.xyxy[0]
                                new_x1, new_y1, new_x2, new_y2 = int(new_x1), int(new_y1), int(new_x2), int(new_y2)
                                new_w, new_h = new_x2 - new_x1, new_y2 - new_y1
                                new_conf = math.ceil(new_box.conf[0] * 100) / 100
                                new_cls = int(new_box.cls[0])
                                new_cls_name = new_labels[new_cls]
        
                                if len(new_boxes) == 1 and new_cls_name == "Helmet":
                                    new_coor.append([new_cls_name, new_x1, new_y1, new_w, new_h])
                                    new_coor.append(["Not Vest"])
                                elif len(new_boxes) == 1 and new_cls_name == "Vest":
                                    new_coor.append(["Not Helmet"])
                                    new_coor.append([new_cls_name, new_x1, new_y1, new_w, new_h])
                                else:
                                    new_coor.append([new_cls_name, new_x1, new_y1, new_w, new_h])
        
                            new_class_coor = set([row[0] for row in new_coor])
        
                            if "Helmet" not in new_class_coor and "Vest" not in new_class_coor:
                                continue
                            elif "Helmet" not in new_class_coor:
                                d[box.id[0].item()].remove("Not Vest")
                                d[box.id[0].item()].insert(2, "Vest")
                            elif "Vest" not in new_class_coor:
                                d[box.id[0].item()].remove("Not Helmet")
                                d[box.id[0].item()].insert(1, "Helmet")
                            else:
                                d[box.id[0].item()].remove("Not Vest")
                                d[box.id[0].item()].remove("Not Helmet")
                                d[box.id[0].item()].insert(1, "Helmet")
                                d[box.id[0].item()].insert(2, "Vest")
                                
                        d1 = d.copy()
                    
                        for i in d:
                            if i not in ids:
                                del d1[i]
                        
                        d = d1.copy()
                        
                        write_list = []
                        for i in d1:
                            write_list.append([d[i][0], d[i][1], d[i][2]])

                        table = pd.DataFrame(write_list, columns=["Id", "Helmet Detection", "Vest Detection"])
                        table["Helmet Detection"] = np.where(table["Helmet Detection"].isin(["Helmet", "Vest"]), "✅", "❌")
                        table["Vest Detection"] = np.where(table["Vest Detection"].isin(["Helmet", "Vest"]), "✅", "❌")
                        result_placeholder.dataframe(table, use_container_width=True,hide_index=True)
                        max_id = box.id[0]
        
                    d[box.id[0].item()][3], d[box.id[0].item()][4], d[box.id[0].item()][5], d[box.id[0].item()][6] = x1, y1, x2, y2
        
        for i in d:
            x1, y1, x2, y2 = d[i][3], d[i][4], d[i][5], d[i][6]
            w, h = x2 - x1, y2 - y1
            if "Helmet" not in d[i] and "Vest" not in d[i]:
                cvzone.putTextRect(img, f"Id: {d[i][0]} NA", (max(0, x1), max(35, y1)), scale=1.2, thickness=1, offset=3, colorR=(255, 0, 0), colorT=(255, 255, 255))
                cvzone.cornerRect(img, (x1, y1, w, h), l=9, colorC=(255, 0, 0))
            elif "Helmet" not in d[i]:
                cvzone.putTextRect(img, f"Id: {d[i][0]} V", (max(0, x1), max(35, y1)), scale=1.2, thickness=1, offset=3, colorR=(255, 0, 0), colorT=(255, 255, 255))
                cvzone.cornerRect(img, (x1, y1, w, h), l=9, colorC=(255, 0, 0))
            elif "Vest" not in d[i]:
                cvzone.putTextRect(img, f"Id: {d[i][0]} H", (max(0, x1), max(35, y1)), scale=1.2, thickness=1, offset=3, colorR=(255, 0, 0), colorT=(255, 255, 255))
                cvzone.cornerRect(img, (x1, y1, w, h), l=9, colorC=(255, 0, 0))
            else:
                cvzone.putTextRect(img, f"Id: {d[i][0]} H V", (max(0, x1), max(35, y1)), scale=1.2, thickness=1, offset=3, colorR=(255, 0, 0), colorT=(255, 255, 255))
                cvzone.cornerRect(img, (x1, y1, w, h), l=9, colorC=(255, 0, 0))
        
        frame_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)

    cap.release()

streamlitApp()