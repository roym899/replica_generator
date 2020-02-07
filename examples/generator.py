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

    def generate(self, out_folder):
        settings = {}
        print(out_folder)
        settings["scene"] = os.path.join(self._dataset_path, self._scenes[1], "habitat", "mesh_semantic.ply")
        settings['width'] = 320
        settings['height'] = 240
        settings["sensor_height"] = 1.5
        settings["color_sensor"] = True
        settings["depth_sensor"] = True
        settings["semantic_sensor"] = True
        settings["silent"] = True
        cfg = make_cfg(settings)
        simulator = habitat_sim.Simulator(cfg)
        
        current_frame = 0
        
        # TODO: randomize position
        
        # do the actual rendering
        observations = simulator.get_sensor_observations()
        self.save_observations(observations, current_frame, out_folder)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_folder", type=str, help="Folder containing Replica dataset")
    parser.add_argument("--output", type=str, help="Output folder", default="")
    args = parser.parse_args()

    generator = Generator(path=args.dataset_folder)
    generator.generate(out_folder=args.output)
    
if __name__ == "__main__":
    main()