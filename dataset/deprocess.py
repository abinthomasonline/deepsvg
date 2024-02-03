import argparse
import os
import pickle
from PIL import Image
from tqdm import tqdm
from deepsvg.svglib.geom import Bbox
from deepsvg.svglib.svg import SVG


def main(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    for file in tqdm(os.listdir(input_folder)):
        if file.endswith(".pkl"):
            with open(os.path.join(input_folder, file), "rb") as f:
                data = pickle.load(f)
            tensor, fillings = data["tensors"][0], data["fillings"]
            svg = SVG.from_tensors(tensor, viewbox=Bbox(256))
            for _svg_path_group, _filling in zip(svg.svg_path_groups, fillings):
                _svg_path_group.path.filling = _filling
            image = svg.draw(do_display=False, return_png=True)
            white_bg = Image.new("RGBA", image.size, "WHITE")
            white_bg.paste(image, (0, 0), image)
            image = white_bg.convert("L")
            image.save(os.path.join(output_folder, file.replace(".pkl", ".jpg")))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_folder', default=os.path.join("dataset", "icons_tensor"))
    parser.add_argument('--output_folder', default=os.path.join("dataset", "icons_jpg"))
    args = parser.parse_args()
    
    main(args.input_folder, args.output_folder)
