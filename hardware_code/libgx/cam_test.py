import pyrealsense2 as rs
import numpy as np
import cv2

# Define the camera serial numbers
camera_serial_1 = "204222061868"
camera_serial_2 = "204322062142"

# Configure depth and color streams for the first camera
pipeline_1 = rs.pipeline()
config_1 = rs.config()
config_1.enable_device(camera_serial_1)
config_1.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config_1.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Configure depth and color streams for the second camera
pipeline_2 = rs.pipeline()
config_2 = rs.config()
config_2.enable_device(camera_serial_2)
config_2.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config_2.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
pipeline_1.start(config_1)
pipeline_2.start(config_2)

try:
    while True:
        # Wait for a coherent pair of frames: depth and color from both cameras
        frames_1 = pipeline_1.wait_for_frames(timeout_ms=10000)  # Increase timeout to 10 seconds
        frames_2 = pipeline_2.wait_for_frames(timeout_ms=10000)  # Increase timeout to 10 seconds

        depth_frame_1 = frames_1.get_depth_frame()
        color_frame_1 = frames_1.get_color_frame()
        depth_frame_2 = frames_2.get_depth_frame()
        color_frame_2 = frames_2.get_color_frame()
       
        if not depth_frame_1 or not color_frame_1 or not depth_frame_2 or not color_frame_2:
            print("One or more frames are missing, skipping this iteration.")
            continue

        # Convert images to numpy arrays
        depth_image_1 = np.asanyarray(depth_frame_1.get_data())
        color_image_1 = np.asanyarray(color_frame_1.get_data())
        depth_image_2 = np.asanyarray(depth_frame_2.get_data())
        color_image_2 = np.asanyarray(color_frame_2.get_data())
        
        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        depth_colormap_1 = cv2.applyColorMap(cv2.convertScaleAbs(depth_image_1, alpha=0.03), cv2.COLORMAP_JET)
        depth_colormap_2 = cv2.applyColorMap(cv2.convertScaleAbs(depth_image_2, alpha=0.03), cv2.COLORMAP_JET)
       
        # Stack both images horizontally for each camera
        images_1 = np.hstack((color_image_1, depth_colormap_1))
        images_2 = np.hstack((color_image_2, depth_colormap_2))

        # Stack the images vertically to display both camera streams
        combined_images = np.vstack((images_1, images_2))

        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', combined_images)

        key = cv2.waitKey(1)
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break

finally:
    # Stop streaming
    pipeline_1.stop()
    pipeline_2.stop()
