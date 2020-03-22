# Replica Generator
Generator for Replica dataset using habitat-sim.

This generates a dataset in [COCO panoptic format](http://cocodataset.org/#format-data).

## Details

Run the script using:
```bash
python3 generator.py --output <output_folder> <replica_v1_folder>
```
This will create the following folder structure / files:
```
<output_folder>/annotations/panoptic_{train,val,test}/*.png
<output_folder>/annotations/panoptic_{train,val,test}.json
<output_folder>/depth/{train,val,test}/*.png
<output_folder>/images/{train,val,test}/*.png
```

Corresponding depth, rgb and panoptic annotation images all have the same filename (e.g., 00001.png). Note that the image count starts from 0 for each train, val, test. Thus, only a combination of folder and image name uniquely identifies an image. 

The semantic segmentation is provided through a combination of a png file (i.e., `<output_folder>/annotations/panoptic_{train,val,test}/*.png`) and json file (i.e., `<output_folder>/annotations/panoptic_{train,val,test}.json`). The png file can be loaded for example using PIL:
```python
import imageio
im = imageio.imread('panoptic_train/00001.png')
```
The images are single channel uint16, where each pixel identifies one object in the scene as in the original replica dataset. The corresponding category of a pixel can be obtained from the corresponding json script. I.e.,:
```python
import json
with open('panoptic_train.json') as f:
    panoptic_json = json.load(f)
image_id = panoptic_json['images'][image_list_id]['id']
category = next(si['category_id'] for si in panoptic_json['annotations'][image_id]['segments_info'] if si['id'] == segment_id) 
```
To summarize each image has a name and a position in the `panoptic_json['images']` list. For this generator the `image_list_id` and `image_id` match, but this is not enforced by COCO format and should not be relied on. Hence, we should obtain the `'id'` from the dictionary first. The pixel value in the panoptic annotation corresponds to the segment_id in the above code.

There are two additional fields in panoptic_{train,val,test}.json. 
```python
panoptic_json['images'][image_list_id]['pose']
```
is a 7 element list, describing the camera pose, as `x,y,z,quat_w,quat_x,quat_y,quat_z`.

```python
panoptic_json['images'][image_list_id]['scene']
```
is a string that uniquely identifies the scene from which an image has been generated. This allows to identify the same object in different images, because the `segment_id` for the same object in different images is the same for a given scene. I.e., to check if an object appears in two different images both `segment_id` and `panoptic_json['images'][image_id]['scene']` has to match.