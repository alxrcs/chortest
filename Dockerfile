#########################################################################
# Dockerfile for testing and development of ChorGram.
# 
# Install Docker for your platform from https://docs.docker.com/install/
#
# Build the image with:
#
#   $ docker build -t chorgram .
# 
# The execution of the previous command may take a while (it is
# downloading and installing libraries, external tools, etc.).
#
# If the above command returns an error, you may want to execute
#
#   $ sudo adduser <your_user_ID> docker
#
# log out and then re-login (this allows non-sudo users to run Docker)
#
# To open a shell with the toolchain you can use:
#
#   $ docker run -v $PWD:/chorgram --rm -it chorgram bash
# 
# Using the GUI from a container is a little bit more involved, but
# you can try with the following.
# 
# Linux
# -----
# docker run --rm -it \ 
#     -v $PWD:/chorgram \ 
#     -v /tmp/.X11-unix:/tmp/.X11-unix \
#     -e DISPLAY=$DISPLAY \
#     chorgram python3 cc/gui.py
# 
# MacOS
# -----
# Install XQuartz from https://www.xquartz.org/
# You need to set the DISPLAY environment variable 
# to correctly point to your X Server.
# 
# docker run --rm -it \ 
#     -v $PWD:/chorgram \
#     -e DISPLAY=$(ipconfig getifaddr en0):0 \ 
#     chorgram python3 cc/gui.py
# 
# In this case, you might need to unrestrict access to the X Server
# The simplest way (though not the most secure) might be just running
# `xhost +`
# 
#################################################################

FROM haskell:8 as build

# Haskell libs
RUN cabal update \
   && cabal install --lib MissingH hxt \
   && cabal install happy

ADD . /chortest
WORKDIR /chortest/chortest/chorgram

RUN make

FROM python:3.11

COPY poetry.lock pyproject.toml /chortest/
COPY . /chortest
COPY --from=build /chortest/chortest/chorgram /chortest/chortest/chorgram

WORKDIR /chortest

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
   && apt-get -y install --no-install-recommends \
   # python3 \
   # python3-dev \
   # python3-pip \
   # python3-setuptools \
   libgraphviz-dev \
   && apt-get autoremove -y \
   && apt-get clean -y \
   && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
RUN update-ca-certificates

RUN pip install "poetry==1.6.1"
RUN poetry config virtualenvs.create false
RUN poetry install --no-interaction --no-ansi

ENV DEBIAN_FRONTEND=dialog

# Extend path
ENV PATH="/chortest:/chortest/chorgram:${PATH}"
CMD /bin/bash