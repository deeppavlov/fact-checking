FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel

RUN apt-get -y update
RUN apt-get install -y git build-essential zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev wget llvm \
    libncurses5-dev libncursesw5-dev xz-utils libffi-dev liblzma-dev

ARG SRC_DIR

ARG PORT
ENV PORT ${PORT}

COPY ./fact_checking/requirements.txt /src/requirements.txt
RUN pip install --upgrade pip setuptools wheel
RUN pip install pybind11==2.2.4
RUN pip install -r /src/requirements.txt

RUN python -c 'import sys; print(sys.version_info[:])' 
COPY $SRC_DIR /src
WORKDIR /src

CMD python server.py
# gunicorn --workers=1 --timeout 500 server:app -b 0.0.0.0:${PORT}