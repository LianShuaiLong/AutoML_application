# description
This is an instance to introduce train xgb with automl framework NNI

### how to start
### prepare data
You should get your data prepared in 'csv' format,

like:

feature1 feature2 ... featureN label
   .        .  
   .        .
   .        .
### begin train
just
``` 
sh start.sh 
```
to begin train

you can change your search space in search_space.json

you can change your trainning setting in config.yml
### stop train
```
sh stop.sh 
```
to end train

the port you used in your experiment will be closed in 10 secs
