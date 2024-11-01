import cv2
import numpy as np
import h5py
import cv2
import pandas as pd
import os
                
def openface_h5(video_path, landmark_path, h5_path, store_size=128):
    """
    crop face from OpenFace landmarks and save a video as .h5 file.

    video_path: the face video path
    landmark_path: landmark .csv file generated by OpenFace.
    h5_path: the path to save the h5_file
    store_size: the cropped face is resized to 128 (default).
    """

    landmark = pd.read_csv(landmark_path)

    with h5py.File(h5_path, 'w') as f:

        total_num_frame = len(landmark)

        cap = cv2.VideoCapture(video_path)
        
        for frame_num in range(total_num_frame):

            if landmark['success'][frame_num]:

                lm_x = []
                lm_y = []
                for lm_num in range(68):
                    lm_x.append(landmark['x_%d'%lm_num][frame_num])
                    lm_y.append(landmark['y_%d'%lm_num][frame_num])

                lm_x = np.array(lm_x)
                lm_y = np.array(lm_y)

                minx = np.min(lm_x)
                maxx = np.max(lm_x)
                miny = np.min(lm_y)
                maxy = np.max(lm_y)

                y_range_ext = (maxy-miny)*0.2
                miny = miny - y_range_ext


                cnt_x = np.round((minx+maxx)/2).astype('int')
                cnt_y = np.round((maxy+miny)/2).astype('int')
                
                break

        bbox_size=np.round(1.0 * (maxy-miny)).astype('int')
        
        ########### init dataset in h5 ##################
        if store_size==None:
            store_size = bbox_size
            
        imgs = f.create_dataset('imgs', shape=(total_num_frame, store_size, store_size, 3), 
                                        dtype='uint8', chunks=(1,store_size, store_size,3),
                                        compression="gzip", compression_opts=4)


        for frame_num in range(total_num_frame):

            if landmark['success'][frame_num]:

                lm_x_ = []
                lm_y_ = []
                for lm_num in range(68):
                    lm_x_.append(landmark['x_%d'%lm_num][frame_num])
                    lm_y_.append(landmark['y_%d'%lm_num][frame_num])

                lm_x_ = np.array(lm_x_)
                lm_y_ = np.array(lm_y_)
                
                lm_x = 0.9*lm_x+0.1*lm_x_
                lm_y = 0.9*lm_y+0.1*lm_y_
                
                minx = np.min(lm_x)
                maxx = np.max(lm_x)
                miny = np.min(lm_y)
                maxy = np.max(lm_y)

                y_range_ext = (maxy-miny)*0.2
                miny = miny - y_range_ext


                cnt_x = np.round((minx+maxx)/2).astype('int')
                cnt_y = np.round((maxy+miny)/2).astype('int')
                
            ret, frame = cap.read()
            # 将每一帧图像从BGR转化成RGB格式
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break
            
            ########## for bbox ################
            bbox_half_size = int(bbox_size/2)
            # print(f"----frame_num: {frame_num}, bbox_size: {bbox_size}, cnt_x: {cnt_x}, cnt_y: {cnt_y}")
            # print(f"---------------frame.shape: {frame.shape}")
            # 按照第一帧所截取的bbox_size的大小，去对后面的帧去做截取的操作（即按照bbox_size的大小去截取对应长度的帧）
            face = np.take(frame, range(cnt_y-bbox_half_size, cnt_y-bbox_half_size+bbox_size),0, mode='clip')
            # print(f"---------------frame.shape: {frame.shape}, face.shape: {face.shape}")
            face = np.take(face, range(cnt_x-bbox_half_size, cnt_x-bbox_half_size+bbox_size),1, mode='clip')
            # print(f"---------------frame.shape: {frame.shape}, face.shape: {face.shape}")
            if store_size==bbox_size:
                imgs[frame_num] = face
            else:
                imgs[frame_num] = cv2.resize(face, (store_size,store_size))

        cap.release()

