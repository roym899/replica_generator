# Author: Leonard Bruns (2020)
"""Script to generate classical computer vision dataset from Replica meshes.
"""

import argparse
import datetime
import json
import os

import numpy as np
from PIL import Image

import habitat_sim
import habitat_sim.agent
from habitat_sim.utils.common import d3_40_colors_rgb
from habitat_sim.agent import AgentState
from habitat_sim.utils.common import quat_from_angle_axis
from habitat_sim.logging import logger
import quaternion

from settings import make_cfg

def create_panoptic_dict():
    panoptic_dict = {}
    now = datetime.datetime.now()
    panoptic_dict['info'] = {'description': 'Replica Generated Dataset',
                             'url': 'https://github.com/roym899/replica_generator', 
                             'version': '1.0', 
                             'year': now.year, 
                             'contributor': '', 
                             'date_created': now.strftime('%Y-%m-%d %H:%M:%S.0')}
    
    panoptic_dict['licenses'] = [] # TODO?
    panoptic_dict['images'] = []
    panoptic_dict['annotations'] = []
    panoptic_dict['categories'] = []
    return panoptic_dict

def create_room(x_1, y_1, z_1, x_2, y_2, z_2):
    x_min = min(x_1, x_2)
    x_max = max(x_1, x_2)
    y_min = min(y_1, y_2)
    y_max = max(y_1, y_2)
    z_min = min(z_1, z_2)
    z_max = max(z_1, z_2)
    return {'x_min': x_min, 
            'x_max': x_max,
            'y_min': y_min,
            'y_max': y_max,
            'z_min': z_min,
            'z_max': z_max}

class Generator:
    """Generator for replica dataset, rgb, depth, and semantics.
    """
    def __init__(self, path):
        self._dataset_path = os.path.normpath(path)

        self._scenes = ["apartment_0", "apartment_1", "apartment_2",
                        "frl_apartment_0", "frl_apartment_1", "frl_apartment_2",
                        "frl_apartment_3", "frl_apartment_4", "frl_apartment_5",
                        "hotel_0", "office_0", "office_1", "office_2",
                        "office_3", "office_4", "room_0", "room_1", "room_2"]
        
        self._scenes = ["apartment_1"]
        
        self._height = 240
        self._width = 320
        
        self._last_frame = None
        self._last_depth_frame = None
        self._last_semantic_frame = None
        
        self._scene_to_rooms = {
            "apartment_0": [
                create_room(-0.3, 1.4, 2.8, 1.68, 0.28, 3.04), # bedroom
                create_room(2.25, -1.38, 3.12, 3.52, -2.0, 2.19), # bathroom
                create_room(-1.65, -2.84, 3.36, -0.08, -5.12, 2.36), # bedroom
                create_room(0.86, -6.93, 2.93, 3.0, -7.85, 2.39), # bedroom
                create_room(2.53, -3.9, 3.15, 3.19, -2.98, 2.81), #bathroom
                create_room(0.95, -3.20, 3.09, 1.76, -4.96, 2.10), #corridor
                create_room(3.92, -4.79, 2.23, 5.07, -4.90, 1.30), #staircase
                create_room(3.7, -8.03, 0.69, 3.5, -6.28, -0.04), #entrance hall
                create_room(1.44, -8.22, 0.37, -1.75, -2.88, -0.27), #livingroom
                create_room(0.12, -0.65, 0.48, 0.89, 1.43, -0.39), #livingroom
                create_room(2.21, 1.98, 0.00, 3.43, 3.63, -0.31), #small table room
                create_room(4.16, 0.07, 0.09, 2.96, -0.24, -0.17), #bathroom
                create_room(4.25, -1.24, 0.52, 2.32, -3.73, -0.46) # workroom
            ],
            "apartment_1": [
                create_room(-1.20, 0.42, 0.53, 0.40, 5.59, -0.22), # livingroom
                create_room(2.76, 5.69, 0.49, 6.5, 4.1, -0.22) # meetingroom
            ],
            "apartment_2": [
                create_room(-1.25, 1.03, 0.79, -0.09, -0.94, -0.26), # workroom
                create_room(1.90, -0.76, 0.35, 5.84, 0.67, -0.50), # bedroom
                create_room(5.69, 3.0, 0.38, 3.97, 4.08, -0.32), # livingroom
                create_room(3.11, 5.86, 0.38, 5.35, 7.75, 0.04) # eatingroom
            ], 
            "frl_apartment_0": [
                create_room(0.55, -4.08, 0.31, 4.58, -1.98, -0.24),
                create_room(4.89, -5.53, -0.31, 3.24, -7.87, 0.36),
                create_room(0.019, 1.89, 0.77, 0.51, 0.45, -0.29)
            ],
            "frl_apartment_1": [
                create_room(-1.96, 0.09, -0.66, -0.79, 0.65, 0.46),
                create_room(1.27, 0.56, 0.78, 3.74, 4.14, 0.04),
                create_room(5.45, 4.85, 0.06, 7.45, 3.51, 0.53)
            ],
            "frl_apartment_2": [
                create_room(-2.1, 0.18, -0.11, -1.09, 0.58, 0.37),
                create_room(1.51, 0.42, 0.37, 3.53, 4.10, 0.83),
                create_room(4.98, 4.68, 0.71, 7.52, 3.21, 0.93)
            ],
            "frl_apartment_3": [
                create_room(-1.79, 0.19, 0.04, 0.20, 0.83, 0.60),
                create_room(1.40, 0.72, 0.55, 4.27, 4.15, 0.90),
                create_room(5.12, 4.69, 0.8, 7.36, 3.45, 0.15)
            ],
            "frl_apartment_4": [
                create_room(-0.07, 2.4, 0.04, 0.08, 0.74, 0.39),
                create_room(0.61, -1.33, 0.55, 3.9, -3.68, -0.04),
                create_room(5.06, -4.87, 0.4, 4.06, -7.39, -0.18)
            ],
            "frl_apartment_5": [
                create_room(-2.19, 0.20, 0.57, 0.17, 0.71, -0.10),
                create_room(1.2, 0.2, 0.57, 4.2, 3.97, 0.15),
                create_room(5.51, 4.35, 0.37, 7.67, 3.04, 0.08)
            ],
            "hotel_0": [
                create_room(-1.56, -0.26, 1.08, 1.4, 0.95, 0.113),
                create_room(2.64, 1.4, 1.0, 4.51, 0.98, -0.07),
                create_room(4.71, -0.15, 0.90, 3.69, -0.65, 0.34)
            ],
            "office_0": [
                create_room(-0.98, 0.36, 1.21, 0.85, -1.56, 0.27)
            ],
            "office_1": [
                create_room(-0.12, -0.50, 1.37, 0.64, 0.93, 0.30)
            ],
            "office_2": [
                create_room(-0.25, -0.76, 1.05, 0.11, 3.26, 0.44)
            ],
            "office_3": [
                create_room(-1.32, -3.80, 1.13, -0.29, 1.14, 0.22)
            ],
            "office_4": [
                create_room(0.24, 2.51, 1.40, 3.45, -0.71, 0.40)
            ],
            "room_0": [
                create_room(0.0, 0.00, 0.51, 4.9, 2.03, 0.02)
            ],
            "room_1": [
                create_room(0.26, 0.22, 0.98, -3.82, -0.19, -0.13)
            ],
            "room_2": [
                create_room(0.75, -1.13, 0.02, 4.22, -0.33, -1.15)
            ]
        }
        
    @staticmethod
    def filename_from_frame_number(frame_number):
        return f"{frame_number:05d}.png"
    
    def load_scene_semantic_dict(self, scene):
        with open(os.path.join(self._dataset_path, scene, 'habitat', 'info_semantic.json'), 'r') as f:
            return json.load(f)
        
    def fix_semantic_observation(self, semantic_observation, scene_dict):
        """Set id of negative categories to 0 to conform to COCO format.
        
        Replica *should* have no 0 id by default. If it does this code must probably be adjusted, except if 0 is undefined.
        """
        for id in np.unique(semantic_observation):
            if scene_dict['id_to_label'][id] < 0:
                semantic_observation[semantic_observation==id] = 0
            elif scene_dict['id_to_label'][id] == 0:
                print('Warning: unexpected id 0 occured, considered as unlabeled...')
                print(scene_dict)
                
        return semantic_observation

    def save_color_observation(self, observation, frame_number, out_folder):
        if not os.path.exists(out_folder):
            os.makedirs(out_folder)
        color_observation = observation["color_sensor"]
        color_img = Image.fromarray(color_observation, mode="RGBA")
        color_img.save(os.path.join(out_folder, self.filename_from_frame_number(frame_number)))
        self._last_frame = np.array(color_img)

    def save_semantic_observation(self, observation, frame_number, out_folder, scene_dict):
        if not os.path.exists(out_folder):
            os.makedirs(out_folder)
        semantic_observation = self.fix_semantic_observation(observation["semantic_sensor"], scene_dict)
        semantic_img = Image.new("I", (semantic_observation.shape[1], semantic_observation.shape[0]))
        self._last_semantic_frame = np.array(semantic_img)
        semantic_img.putdata((semantic_observation.flatten()))
        semantic_img.save(os.path.join(out_folder, self.filename_from_frame_number(frame_number)))

    def save_depth_observation(self, observation, frame_number, out_folder):
        if not os.path.exists(out_folder):
            os.makedirs(out_folder)
        depth_observation = observation["depth_sensor"]
        depth_img = Image.fromarray(
            (depth_observation / 10 * 255).astype(np.uint8), mode="L"
        )
        depth_img.save(os.path.join(out_folder, self.filename_from_frame_number(frame_number)))
        self._last_depth_frame = np.array(depth_img)

    def save_observations(self, observation, frame_number, out_folder, split_name, scene_dict):
        self.save_color_observation(observation, frame_number, os.path.join(out_folder, 'images', split_name))
        self.save_semantic_observation(observation, frame_number, os.path.join(out_folder, 'annotation', f"panoptic_{split_name}"), scene_dict)
        self.save_depth_observation(observation, frame_number, os.path.join(out_folder, 'depth', split_name))
        
    def update_dict(self, panoptic_dict, scene_dict, frame_number, out_folder, split_name):
        panoptic_dict['images'].append({
            'file_name': self.filename_from_frame_number(frame_number),
            'height': self._height,
            'width': self._width,
            'id': frame_number
        })
        
        panoptic_dict['annotations'].append({
            'segments_info': [],
            'file_name': self.filename_from_frame_number(frame_number),
            'image_id': frame_number
        })
        
        for id in np.unique(self._last_semantic_frame):
            label = scene_dict['id_to_label'][id]
            mask = self._last_semantic_frame == id
            xs, ys = np.nonzero(mask)
            minx, miny, maxx, maxy = min(xs), min(ys), max(xs), max(ys)
            
            if label != 0: # == 0 would be undefined -> no annotation
                panoptic_dict['annotations'][-1]['segments_info'].append({
                    'id': int(id), # the number in the semantic image
                    'category_id': int(label), # the matching category id 
                    'iscrowd': 0, 
                    'bbox': [int(minx),int(miny),int(maxx-minx),int(maxy-miny)], # x, y, (starting top left) width, height
                    'area': int(np.sum(mask)) # area in pixels (exact, not bounding box)
                })
    
    def save_dict(self, panoptic_dict, out_folder, split_name):
        with open(os.path.join(out_folder, 'annotation', f"panoptic_{split_name}.json"), 'w') as f:
            json.dump(panoptic_dict, f)

    def generate(self, out_folder, split_name, frames_per_room=100):
        """Generates dataset at specified path.
        
        Resulting folder structure (same as COCO)
        {out_folder}/annotation/panoptic_{split_name}/*.png
        {out_folder}/annotation/panoptic_{split_name}.json
        
        {out_folder}/images/{split_name}/*.png
        
        {out_folder}/depth/{split_name}/*.png (this is not in COCO)
        
        Args:
            out_folder: The folder to write the dataset to.
            split_name: Subfolder name, i.e., train, val, test
        """
        settings = {}
        print(out_folder)
        settings['width'] = self._width
        settings['height'] = self._height
        settings["sensor_height"] = 0
        settings["color_sensor"] = True
        settings["depth_sensor"] = True
        settings["semantic_sensor"] = True
        settings["silent"] = True
        
        current_frame = 0
        total_frames = 0
        for scene in self._scenes:
            for room in self._scene_to_rooms[scene]:
                total_frames += frames_per_room
        
        panoptic_dict = create_panoptic_dict()        
        
        for scene in self._scenes:
            # setup the simulator
            settings["scene"] = os.path.join(self._dataset_path, scene, "habitat", "mesh_semantic.ply")
            cfg = make_cfg(settings)
            simulator = habitat_sim.Simulator(cfg)
            
            # load semantic information for scene
            scene_semantic_dict = self.load_scene_semantic_dict(scene)
            
            # generate data for each room
            for room in self._scene_to_rooms[scene]:
                for _ in range(0,frames_per_room):
                    agent = simulator.get_agent(0)
                    agent_state = agent.get_state()
                    random_state = AgentState()
                    
                    # this looks weird because coordinates have been collected in replica viewer,
                    # which uses a different coordinate system than habitat-sim
                    random_state.position[0] = np.random.uniform(room['x_min'], room['x_max'])
                    random_state.position[1] = np.random.uniform(room['z_min'], room['z_max'])
                    random_state.position[2] = - np.random.uniform(room['y_min'], room['y_max'])
                    random_state.rotation = (quat_from_angle_axis(np.random.uniform(0,np.pi), np.array([0,1,0]))*
                                             quat_from_angle_axis(np.random.uniform(-np.pi/3,np.pi/16), np.array([1,0,0]))*
                                             quat_from_angle_axis(np.random.uniform(-np.pi/16,np.pi/16), np.array([0,0,1])))
                    
                    agent_state.sensor_states = {}
                    agent.set_state(random_state)
                    
                    # do the actual rendering
                    observations = simulator.get_sensor_observations()
                    
                    self.save_observations(observations, current_frame, out_folder, split_name, scene_semantic_dict)
                    
                    self.update_dict(panoptic_dict, scene_semantic_dict, current_frame, out_folder, split_name)
                    
                    print(f'Saved image {current_frame+1}/{total_frames}')
                    current_frame += 1
                    
            simulator.close()
            
            del simulator
        
        self.save_dict(panoptic_dict, out_folder, split_name)    
            

def main():
    """Main function of the program.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_folder", type=str, help="Folder containing Replica dataset")
    parser.add_argument("--output", type=str, help="Output folder", default="")
    args = parser.parse_args()

    generator = Generator(path=args.dataset_folder)
    generator.generate(out_folder=args.output, 
                       split_name='train',
                       frames_per_room=6)
    generator.generate(out_folder=args.output, 
                       split_name='val',
                       frames_per_room=2)
    generator.generate(out_folder=args.output, 
                       split_name='test',
                       frames_per_room=2)
    
if __name__ == "__main__":
    main()