FROM ros:noetic

ADD https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc /tmp/ros.asc
RUN apt-key add /tmp/ros.asc
RUN apt-get update -y && apt-get install -y ros-noetic-compressed-image-transport
RUN apt-get update -y && apt-get install -y python3-pip
RUN pip3 install numpy==1.17.5 pygame
RUN pip3 install tensorflow 
RUN pip3 install gym 
RUN pip3 install keras
RUN pip3 install pandas
RUN pip3 install torch
#--no-cache-dir
WORKDIR /root/projects/
RUN echo 'catkin_make install && source /root/projects/devel/setup.bash' >> /root/.bashrc
