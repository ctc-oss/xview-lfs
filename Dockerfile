FROM ctcoss/get-lfs

WORKDIR /opt/src
COPY . /opt/src

RUN python -m pip install -r requirements.txt --no-cache-dir

ENTRYPOINT ["python", "-m", "xview_lfs.yolo"]
