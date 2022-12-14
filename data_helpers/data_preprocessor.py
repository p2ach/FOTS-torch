import os

import numpy as np
import pandas as pd

from torch.utils.data import DataLoader

from data_helpers.datasets import Synth800kDataset, ICDARDataset, AIHubDataset
from data_helpers.data_utils import synth800k_collate, icdar_collate, aihub_collate

from tqdm import tqdm


def aihub_preprocess(root_dir,config):
    """
    Preprocess the data to save GPU time while training because 
    ground truth generation is very time consuming for FOTS.
    """
    # dataset = Synth800kDataset(config["data_dir"])
    # dataset = ICDARDataset(root_dir,config["image_dir"],config["gt_dir"])
    dataset = AIHubDataset(root_dir,config)


    data_loader = DataLoader(
        dataset,
        num_workers=config["num_workers"],
        pin_memory=True,
        batch_size=config["batch_size"],
        shuffle=False,
        collate_fn=aihub_collate
    )

    os.makedirs(os.path.join(root_dir,config["output_dir"], "image"), exist_ok=True)
    os.makedirs(os.path.join(root_dir,config["output_dir"], "score"), exist_ok=True)
    os.makedirs(os.path.join(root_dir,config["output_dir"], "geo"), exist_ok=True)
    os.makedirs(os.path.join(root_dir,config["output_dir"], "training_mask"), exist_ok=True)
    os.makedirs(os.path.join(root_dir,config["output_dir"], "bboxes"), exist_ok=True)
    os.makedirs(os.path.join(root_dir,config["output_dir"], "transcripts"), exist_ok=True)

    img_list, sm_list, gm_list, tm_list, bboxes, transcripts, mappings = [], [], [], [], [], [], []


    for idx, batch in tqdm(enumerate(data_loader), total=len(data_loader), position=0, leave=True):
        if idx > 3450:
            break


        image_paths, images, bboxs, training_masks, texts, score_maps, geo_maps, mapping = batch



        for pth, img, bbox, txt, scr, geo, tm in zip(image_paths, images, bboxs, texts, score_maps, geo_maps, training_masks):
            # img_pth = pth.split("/")[-2:]
            img_name = pth.split(".")[0]
            flag_abnormal = False
            for _tm in tm :
                if _tm is None:
                    flag_abnormal = True

            if not flag_abnormal:
                img_list.append(f"image/{img_name}.npy")
                sm_list.append(f"score/{img_name}_score_map.npy")
                gm_list.append(f"geo/{img_name}_geo_map.npy")
                tm_list.append(f"training_mask/{img_name}_tm.npy")
                bboxes.append(f"bboxes/{img_name}_bboxes.npy")
                transcripts.append(f"transcripts/{img_name}_transcripts.npy")

                if not os.path.isfile(f"image/{img_name}.npy"):
                    np.save(f"{root_dir}{config['output_dir']}/image/{img_name}.npy", img.numpy().astype(np.uint8))
                    np.save(f"{root_dir}{config['output_dir']}/score/{img_name}_score_map.npy", scr.numpy().astype(np.uint8))
                    np.save(f"{root_dir}{config['output_dir']}/geo/{img_name}_geo_map.npy", geo.numpy().astype(np.float32))
                    np.save(f"{root_dir}{config['output_dir']}/training_mask/{img_name}_tm.npy", tm.numpy().astype(np.uint8))
                    np.save(f"{root_dir}{config['output_dir']}/bboxes/{img_name}_bboxes.npy", bbox.astype(np.float32))
                    np.save(f"{root_dir}{config['output_dir']}/transcripts/{img_name}_transcripts.npy", txt)
        
        if idx == config["num_iterations"]:
            break

    data_df = pd.DataFrame({
        "images": img_list,
        "score_maps": sm_list,
        "geo_maps": gm_list,
        "training_masks": tm_list,
        "bboxes": bboxes,
        "transcripts": transcripts
    })
    data_df.to_csv(f"{root_dir}{config['output_dir']}/train.csv", index=False)

    print(f"Generated ground truths for {len(data_df)} images.")

def preprocess(root_dir,config):
    """
    Preprocess the data to save GPU time while training because
    ground truth generation is very time consuming for FOTS.
    """
    # dataset = Synth800kDataset(config["data_dir"])
    dataset = ICDARDataset(root_dir,config["image_dir"],config["gt_dir"])
    # dataset = AIHubDataset(root_dir,config)


    data_loader = DataLoader(
        dataset,
        num_workers=config["num_workers"],
        pin_memory=True,
        batch_size=config["batch_size"],
        shuffle=False,
        collate_fn=icdar_collate
    )

    os.makedirs(os.path.join(root_dir,config["output_dir"], "image"), exist_ok=True)
    os.makedirs(os.path.join(root_dir,config["output_dir"], "score"), exist_ok=True)
    os.makedirs(os.path.join(root_dir,config["output_dir"], "geo"), exist_ok=True)
    os.makedirs(os.path.join(root_dir,config["output_dir"], "training_mask"), exist_ok=True)
    os.makedirs(os.path.join(root_dir,config["output_dir"], "bboxes"), exist_ok=True)
    os.makedirs(os.path.join(root_dir,config["output_dir"], "transcripts"), exist_ok=True)

    img_list, sm_list, gm_list, tm_list, bboxes, transcripts, mappings = [], [], [], [], [], [], []


    for idx, batch in tqdm(enumerate(data_loader), total=len(data_loader), position=0, leave=True):

        if idx > 2:
            break


        image_paths, images, bboxs, training_masks, texts, score_maps, geo_maps, mapping = batch



        for pth, img, bbox, txt, scr, geo, tm in zip(image_paths, images, bboxs, texts, score_maps, geo_maps, training_masks):
            # img_pth = pth.split("/")[-2:]
            img_name = pth.split(".")[0]

            img_list.append(f"image/{img_name}.npy")
            sm_list.append(f"score/{img_name}_score_map.npy")
            gm_list.append(f"geo/{img_name}_geo_map.npy")
            tm_list.append(f"training_mask/{img_name}_tm.npy")
            bboxes.append(f"bboxes/{img_name}_bboxes.npy")
            transcripts.append(f"transcripts/{img_name}_transcripts.npy")

            np.save(f"{config['output_dir']}/image/{img_name}.npy", img.numpy().astype(np.uint8))
            np.save(f"{config['output_dir']}/score/{img_name}_score_map.npy", scr.numpy().astype(np.uint8))
            np.save(f"{config['output_dir']}/geo/{img_name}_geo_map.npy", geo.numpy().astype(np.float32))
            np.save(f"{config['output_dir']}/training_mask/{img_name}_tm.npy", tm.numpy().astype(np.uint8))
            np.save(f"{config['output_dir']}/bboxes/{img_name}_bboxes.npy", bbox.astype(np.float32))
            np.save(f"{config['output_dir']}/transcripts/{img_name}_transcripts.npy", txt)

        if idx == config["num_iterations"]:
            break

    data_df = pd.DataFrame({
        "images": img_list,
        "score_maps": sm_list,
        "geo_maps": gm_list,
        "training_masks": tm_list,
        "bboxes": bboxes,
        "transcripts": transcripts
    })
    data_df.to_csv(f"{config['output_dir']}/train.csv", index=False)

    print(f"Generated ground truths for {len(data_df)} images.")
