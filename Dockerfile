FROM gcr.io/tensorflow/tensorflow
RUN pip install pillow
RUN pip install lxml
RUN pip install flask
RUN pip install flask-restful

RUN apt-get update && apt-get install -y --no-install-recommends protobuf-compiler

COPY object_detection/ /object_detection/
COPY server.py /server.py

WORKDIR /

RUN protoc object_detection/protos/*.proto --python_out=.

ENV MODELNAME=ssd_inception_v2_coco_11_06_2017

ADD http://download.tensorflow.org/models/object_detection/$MODELNAME.tar.gz /

RUN mv $MODELNAME model_coco

EXPOSE 5000

CMD python server.py