# TensorFlow Models (REST API wrapper)
Simple REST app for image classification using models pre-trained on the COCO dataset.
If you want to change the model used by the app just change the `MODELNAME` variable in the `Dockerfile`.
List of available pre-trained object detection models can be found under [Tensorflow detection model zoo](https://github.com/tensorflow/models/blob/master/object_detection/g3doc/detection_model_zoo.md)

## Run (Docker)
```
docker build -t coco-model .
docker run -p 5000:5000 coco-model
# test with: curl -v http://localhost:5000/detect?imageUrl=<imageUrl>, e.g.
curl -v http://localhost:5000/detect?imageUrl=http://expressioncoffins.com.au/wp-content/uploads/2012/06/RED-TRACTOR1.jpg
```
You should get the following response:
```
{
   "imageUrl":"http://expressioncoffins.com.au/wp-content/uploads/2012/06/RED-TRACTOR1.jpg",
   "results":[
      {
         "score":0.899250328540802,
         "label":"truck"
      },
      {
         "score":0.582574725151062,
         "label":"person"
      },
      {
         "score":0.0435677208006382,
         "label":"umbrella"
      },
      {
         "score":0.029505163431167603,
         "label":"kite"
      },
      {
         "score":0.021187135949730873,
         "label":"clock"
      }
   ]
}
```

## Run(GCP)
```
docker build -t eu.gcr.io/bw-dev-analytics0/coco-model .
gcloud docker -- push eu.gcr.io/bw-dev-analytics0/coco-model
kubectl run coco-model --image=eu.gcr.io/$PROJECT_NAME/coco-model --port=5000 --replicas=2
kubectl expose deployment coco-model --type=LoadBalancer --name=coco-model
```