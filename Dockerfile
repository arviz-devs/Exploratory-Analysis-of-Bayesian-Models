FROM ruby:2.7.1-buster

RUN apt-get update && apt-get install -y git build-essential pandoc vim ffmpeg \
    dos2unix libgtk-3-dev libdbus-glib-1-dev libxt-dev make python3.7 python3-pip \
    && rm -rf /var/lib/apt/lists/*


RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install -r requirements.txt
RUN python3 -m pip install -r jupyter-book


WORKDIR /opt/eda_bm
COPY . .

RUN gem install bundler:1.17.2
RUN make install
