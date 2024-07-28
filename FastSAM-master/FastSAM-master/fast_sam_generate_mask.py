from fastsam import FastSAM, FastSAMPrompt
import json
import torch
import argparse

from visualize import *

DEVICE='cuda'

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

if __name__ == "__main__":
    # write hyperparameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--training_idx', type=int, default=0)
    args = parser.parse_args()

    i = args.training_idx
    
    glob_scale = 1.0
    base_path = 'scratch/hubmap/processed/cropped_train'
    test_files = sorted(glob.glob(os.path.join(base_path, '*.tiff')))
    model = FastSAM('./weights/FastSAM-x.pt')

    # image_id = get_image_id(train_files[i])
    image_id = get_image_id(train_files[i])
    tiff_images = glob.glob(f"{base_path}/{image_id}/*.tiff")
    skipped_tiles = []
    # find valid tiles
    valid_tiles = json.load(open(f"{base_path}/{image_id}/valid_tiles.json"))
    valid_tiles = [tile_name.split("/")[-1] for tile_name in valid_tiles]
    for tiff_image in tqdm(tiff_images, desc=f"Tiles"):
        tiff_image = "scratch/hubmap/processed/cropped_train/095bf7a1f/tile_25_23.tiff"
        if tiff_image.split("/")[-1] not in valid_tiles:   # skip invalid tiles
            continue
        image, shape = read_image(tiff_image, scale=glob_scale)
        if shape != (1024, 1024, 3):
            skipped_tiles.append(tiff_image)
            continue

        IMAGE_PATH=tiff_image
        everything_results = model(IMAGE_PATH, device=DEVICE, retina_masks=True, imgsz=1024, conf=0.1, iou=0.1,)
        prompt_process = FastSAMPrompt(IMAGE_PATH, everything_results, device=DEVICE)
        # ann = prompt_process.everything_prompt()
        ann = prompt_process.text_prompt(text="large glomerulus cells (kidney), which cells contain dark materials in the middle part.", top_k=30)

        splits = tiff_image.split("/")
        os.makedirs(f'scratch/hubmap/mask/{splits[-2]}', exist_ok=True)
        prompt_process.plot(annotations=ann,output_path=f'scratch/hubmap/mask/{splits[-2]}/{splits[-1].split(".")[0]}.jpg',retina=True)
        # save ann
        torch.save(ann, f"scratch/hubmap/mask/{splits[-2]}/{splits[-1].split('.')[0]}.pt")

        # save the mask to json file
        # json.dump(ann, open(f"{tiff_image.replace('tiff', 'json')}", "w"), cls=NumpyEncoder)
        # print(f"saved {tiff_image.replace('tiff', 'json')}")

    json.dump(skipped_tiles, open(f"{base_path}/{image_id}/skipped_tiles.json", "w"), cls=NumpyEncoder)