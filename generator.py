# Author: Leonard Bruns (2020)
"""Script to generate classical computer vision dataset from Replica meshes.
"""

import argparse
import os

import numpy as np
from PIL import Image
from settings import make_cfg

import habitat_sim
import habitat_sim.agent
from habitat_sim.utils.common import d3_40_colors_rgb

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
                create_room(0.55, -4.08, 0.31, 4.58, -1.98, -0.24),
                create_room(4.89, -5.53, -0.31, 3.24, -7.87, 0.36),
                create_room(0.019, 1.89, 0.77, 0.51, 0.45, -0.29)
                ],
            "frl_apartment_2": [
                create_room(0.55, -4.08, 0.31, 4.58, -1.98, -0.24),
                create_room(4.89, -5.53, -0.31, 3.24, -7.87, 0.36),
                create_room(0.019, 1.89, 0.77, 0.51, 0.45, -0.29)
                ],
            "frl_apartment_3": [
                create_room(0.55, -4.08, 0.31, 4.58, -1.98, -0.24),
                create_room(4.89, -5.53, -0.31, 3.24, -7.87, 0.36),
                create_room(0.019, 1.89, 0.77, 0.51, 0.45, -0.29)
                ],
            "frl_apartment_4": [
                create_room(0.55, -4.08, 0.31, 4.58, -1.98, -0.24),
                create_room(4.89, -5.53, -0.31, 3.24, -7.87, 0.36),
                create_room(0.019, 1.89, 0.77, 0.51, 0.45, -0.29)
                ],
            "frl_apartment_5": [
                create_room(0.55, -4.08, 0.31, 4.58, -1.98, -0.24),
                create_room(4.89, -5.53, -0.31, 3.24, -7.87, 0.36),
                create_room(0.019, 1.89, 0.77, 0.51, 0.45, -0.29)
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

    def save_color_observation(self, observation, frame_number, out_folder):
        color_observation = observation["color_sensor"]
        color_img = Image.fromarray(color_observation, mode="RGBA")
        color_img.save(os.path.join(out_folder, "rgba_%05d.png" % frame_number))

    def save_semantic_observation(self, observation, frame_number, out_folder):
        semantic_observation = observation["semantic_sensor"]
        semantic_img = Image.new("P", (semantic_observation.shape[1], semantic_observation.shape[0]))
        semantic_img.putpalette(d3_40_colors_rgb.flatten())
        semantic_img.putdata((semantic_observation.flatten() % 40).astype(np.uint8))
        semantic_img.save(os.path.join(out_folder, "semantic_%05d.png" % frame_number))

    def save_depth_observation(self, observation, frame_number, out_folder):
        depth_observation = observation["depth_sensor"]
        depth_img = Image.fromarray(
            (depth_observation / 10 * 255).astype(np.uint8), mode="L"
        )
        depth_img.save(os.path.join(out_folder, "depth_%05d.png" % frame_number))

    def save_observations(self, observation, frame_number, out_folder):
        self.save_color_observation(observation, frame_number, out_folder)
        self.save_semantic_observation(observation, frame_number, out_folder)
        self.save_depth_observation(observation, frame_number, out_folder)

    def generate(self, out_folder, frames_per_room=1):
        """Generates dataset at specified path.
        
        Args:
            out_folder: The folder to write the dataset to.
        """
        settings = {}
        print(out_folder)
        settings['width'] = 320
        settings['height'] = 240
        settings["sensor_height"] = 1.5
        settings["color_sensor"] = True
        settings["depth_sensor"] = True
        settings["semantic_sensor"] = True
        settings["silent"] = True
        
        
        current_frame = 0
        
        for scene in self._scenes:
            settings["scene"] = os.path.join(self._dataset_path, scene, "habitat", "mesh_semantic.ply")
            cfg = make_cfg(settings)
            simulator = habitat_sim.Simulator(cfg)
            
            for room in self._scene_to_rooms[scene]:
                for _ in range(0,frames_per_room):
                    # TODO: randomize position
                    
                    # do the actual rendering
                    observations = simulator.get_sensor_observations()
                    
                    self.save_observations(observations, current_frame, out_folder)
                    current_frame += 1
                    
            simulator.close()
            del simulator
            

def main():
    """Main function of the program.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_folder", type=str, help="Folder containing Replica dataset")
    parser.add_argument("--output", type=str, help="Output folder", default="")
    args = parser.parse_args()

    generator = Generator(path=args.dataset_folder)
    generator.generate(out_folder=args.output)
    
if __name__ == "__main__":
    main()