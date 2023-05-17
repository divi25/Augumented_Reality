import cv2
import numpy as np
import os



directory_path = '/home/sandeep/spatial/main'

if os.access(directory_path, os.W_OK):
    print("You have write permission for the directory.")
else:
    print("You do not have write permission for the directory.")


current_directory = os.getcwd()
print(current_directory)

xyz = 'home/sandeep/spatial/main/img1.png'

target_image = cv2.imread('image_1.jpg')
cv2.imshow('ma',target_image)
cv2.waitKey(0)

if target_image is None:
    print(f"Failed to load the target image from '{xyz}' ")

# Create a feature detector
orb = cv2.ORB_create()
keypoints = orb.detect(target_image, None)

# Draw keypoints on the image
image_with_keypoints = cv2.drawKeypoints(target_image, keypoints, None, color=(0, 255, 0), flags=0)

# Display the image with keypoints
cv2.imshow('Image with Keypoints', image_with_keypoints)
cv2.waitKey(0)

# Find keypoints and descriptors in the target image
kp_target, des_target = orb.detectAndCompute(target_image, None)

# Create a brute-force matcher
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Create a perspective cube for overlay
cube_points_3d = np.float64([[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0], [0, 0, -1], [0, 1, -1], [1, 1, -1], [1, 0, -1]])
cube_points_2d = np.float64([[0, 0], [0, 300], [300, 300], [300, 0], [0, 0], [0, 300], [300, 300], [300, 0]])

# Read the video file
video_feed = cv2.VideoCapture('Div.mp4')

output_width, output_height = int(video_feed.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video_feed.get(cv2.CAP_PROP_FRAME_HEIGHT))
output_video = cv2.VideoWriter('output_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (output_width, output_height))


while True:
    # Read a frame from the video feed
    ret, frame = video_feed.read()
    if not ret:
        break
    
    # Find keypoints and descriptors in the current frame
    kp_frame, des_frame = orb.detectAndCompute(frame, None)
    
    # Match descriptors between the target image and current frame
    matches = bf.match(des_target, des_frame)
    
    # Sort matches by distance
    matches = sorted(matches, key=lambda x: x.distance)
    
    # Take top matches (you can adjust this threshold)
    good_matches = matches[:8]  # Adjust the threshold as needed
    
    # Check if enough valid matches are available
    if len(good_matches) < 4:
        continue
    
    # Extract matched keypoints
    src_pts = np.float64([kp_target[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.array([kp_frame[m.trainIdx].pt for m in good_matches], dtype=np.float64).reshape(-1, 1, 2)
    
     # print(len(dst_pts))
    #print(len(src_pts))
   # print(len(cube_points_3d))
    #print(dst_pts.shape)
 #   print(src_pts.dtype)
  #  print(dst_pts.dtype)
    

    cube_points_3d = cube_points_3d.reshape((8, 1, 3))
    dst_pts = dst_pts.reshape((8, 1, 2))

 # Ensure camera_matrix and dist_coeffs are properly defined
    camera_matrix = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]], dtype=np.float64)
    dist_coeffs = np.array([0,0,0,0], dtype=np.float64)
#homography matrix usin ransac
    M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    #print (M)
# Calculate the pose of the target
    _, rvec, tvec = cv2.solvePnP(cube_points_3d, dst_pts, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
    #print(str(cube_points_3d),str(dst_pts))

  

    
    # Project 3D points of the cube onto the image plane
    cube_points_2d_proj, _ = cv2.projectPoints(cube_points_3d, rvec, tvec,camera_matrix, dist_coeffs)
    
      # Draw the wireframe cube on the image
    frame = cv2.polylines(frame, [np.int32(cube_points_2d_proj)], True, (0, 255, 0), 3)
    
    # Show the resulting frame
    cv2.imshow('AR Markerless Tracking', frame)
    output_video.write(frame)
    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video feed and close all windows
video_feed.release()
cv2.destroyAllWindows()

