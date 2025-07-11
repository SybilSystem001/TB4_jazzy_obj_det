#Install dependencies
```
sudo apt install ros-jazzy-rqt-image-view
sudo apt install ros-jazzy-depthai-ros 
sudo apt install ros-jazzy-cv-bridge ros-jazzy-image-transport python3-opencv
sudo apt install ros-jazzy-vision-msgs ros-jazzy-image-transport
sudo apt install ros-jazzy-depthai-ros ros-jazzy-movei ros-jazzy-nav2-bringup
```
#clone the repo in your src folder and then build the package
```
git clone -b jazzy https://github.com/SybilSystem001/TB4_jazzy_obj_det.git
```
```
colcon build --packages-select yolov4_overlay
```

#run the package with the following command
```ROS2
ros2 run yolov4_overlay yolov4_overlay
```
