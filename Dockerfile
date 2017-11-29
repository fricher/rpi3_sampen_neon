FROM arm64v8/debian:latest
RUN apt-get update
RUN apt-get install -y git
RUN apt-get install -y cmake
RUN apt-get install -y g++
RUN git clone https://github.com/fricher/rpi3_sampen_neon
WORKDIR rpi3_sampen_neon/build
RUN cmake .. || true
RUN cmake --build .
