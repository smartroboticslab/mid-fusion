FROM ubuntu:18.04

ARG DEBIAN_FRONTEND=noninteractive

# system
RUN echo "Installing apt packages..." \
	&& export DEBIAN_FRONTEND=noninteractive \
	&& apt -y update --no-install-recommends \
	&& apt -y install --no-install-recommends \
	git \
	wget \
    libeigen3-dev \
    freeglut3-dev \
	sudo \
	&& apt autoremove -y \
	&& apt clean -y \
	&& export DEBIAN_FRONTEND=dialog

RUN DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get -y install tzdata

# install OKVIS dependencies
RUN echo "Installing apt packages..." \
    && export DEBIAN_FRONTEND=noninteractive \
    && apt -y update --no-install-recommends \
    && apt -y install --no-install-recommends \
    cmake \
    # libatlas-base-dev \
    libeigen3-dev \
    # libsuitesparse-dev \
    libeigen3-dev libopencv-dev qtbase5-dev \
    gcc-6 g++-6 \
    build-essential \
    libboost-all-dev \
    libopencv-dev \
    # libceres-dev \
    libomp-dev \
    ca-certificates \
    && apt autoremove -y \
    && apt clean -y \
    && export DEBIAN_FRONTEND=dialog

# # install librealsense
# RUN echo "Installing librealsense..." \
#     && export DEBIAN_FRONTEND=noninteractive \
#     && apt -y update --no-install-recommends \
#     && apt -y install --no-install-recommends \
#     libusb-1.0-0-dev \
#     libglfw3-dev \
#     pkg-config \
#     libgtk-3-dev \
#     gcc-6 g++-6 \
#     build-essential \
#     && apt autoremove -y \
#     && apt clean -y \
#     && export DEBIAN_FRONTEND=dialog

# # install legacy realsense 1
# COPY ./thirdparty/librealsense /opt/librealsense
# RUN cd /opt/librealsense \
#     && mkdir build \
#     && cd build \
#     && cmake -DCMAKE_BUILD_TYPE=Release .. \
#     && make -j8 \
#     && make install
    
# # install geographiclib
# RUN echo "Installing geographiclib..." 
# RUN git clone git://git.code.sourceforge.net/p/geographiclib/code /opt/geographiclib \
#     && cd /opt/geographiclib \
#     && mkdir build \
#     && cd build \
#     && cmake -DCMAKE_BUILD_TYPE=Release .. \
#     && make -j8 \
#     && make install


# # RUN git clone git@bitbucket.org:sleutenegger/okvis.git /opt/okvis \
# # RUN cd /opt/okvis \
#     # && git checkout ofusion++ \
# # TODO: hack, because ofusion++ branch is not public -> replace it with public branch later
# COPY ./thirdparty/okvis /opt/okvis
# RUN cd /opt/okvis \
#     && mkdir build \
#     && cd build \
#     && cmake -D CMAKE_C_COMPILER=gcc-6 -D CMAKE_CXX_COMPILER=g++-6 -DCMAKE_BUILD_TYPE=Release .. \
#     && make -j8 \
#     && make install

# install librealsense
RUN echo "Installing GUI stuff..." \
    && export DEBIAN_FRONTEND=noninteractive \
    && apt -y update --no-install-recommends \
    && apt -y install --no-install-recommends \
    language-pack-en-base \
    libcanberra-gtk-module libcanberra-gtk3-module \
    && apt autoremove -y \
    && apt clean -y \
    && export DEBIAN_FRONTEND=dialog
RUN dpkg-reconfigure locales && \
    locale-gen en_US.UTF-8 && \
    /usr/sbin/update-locale LANG=en_US.UTF-8
