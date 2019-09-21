#!/usr/bin/env python
# coding: utf-8
import argparse
import copy
import glob
import os
import shutil

from utils import get_config, pytorch03_to_pytorch04

from torch.autograd import Variable
import torchvision.utils as vutils
import torch
from torchvision import transforms
from trainer import MUNIT_Trainer
from glob import glob

from functools import partial
import json
from PIL import Image
import numpy as np
import tqdm
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--synthetic-folder", required=True)
    parser.add_argument("--real-glob", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split_names", default=["train", "test", "val"], nargs='+')
    parser.add_argument("--blocksize", default=8, type=int)
    parser.add_argument("--blockidx", default=0, type=int)
    parser.add_argument("--nvar", default=5, type=int)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--in-width", default=908, type=int, help="If -1, don't resize") 
    parser.add_argument("--in-height", default=512, type=int, help="If -1, don't resize")
    parser.add_argument("--out-width", default=1280, type=int)
    parser.add_argument("--out-height", default=720, type=int)
    parser.add_argument("--format", default='yolo')
    return parser.parse_args()


class Options(object):
    def __init__(self, config, checkpoint):
        self.config = config
        self.output_folder = ''
        self.checkpoint = checkpoint
        self.style = ''
        self.a2b = 1
        self.seed = 10
        self.num_style = 10
        self.synchronized = ''
        self.output_only = ''
        self.output_path = '.'
        self.trainer = 'MUNIT'
    

def runner(args, partition):
    synth_path = Path(args.synthetic_folder)

    ann_file = f"{synth_path / 'annotations/instances_{}.json'}".format(partition)
    with open(ann_file, "r") as f:
        ann = json.load(f)

    if args.format == 'yolo':
        synthetic_images_list_file = f"{synth_path / 'annotations/{}.txt'}".format(partition)
        synthetic_labels_base_path = f"{synth_path / 'labels/{}/'}".format(partition)
        with open(synthetic_images_list_file, "r") as f:
            synthetic_images_list = sorted(list([f.replace("\n", "") for f in f.readlines()]))
    elif args.format == 'coco':
        synthetic_images_list = list(sorted(img['path'] for img in ann['images']))

    real_images_list = sorted(list(glob(args.real_glob)))

    opts = Options(args.config, args.checkpoint)
    
    torch.manual_seed(opts.seed)
    torch.cuda.manual_seed(opts.seed)
    
    # Load experiment setting
    config = get_config(opts.config)
    opts.num_style = 1 if opts.style != '' else opts.num_style
    
    
    if 'new_size' in config:
        new_size = config['new_size']
    else:
        if opts.a2b==1:
            new_size = config['new_size_a']
        else:
            new_size = config['new_size_b']
    
    def load_checkpoint(checkpoint):
        # Setup model and data loader
        config['vgg_model_path'] = opts.output_path
    
        style_dim = config['gen']['style_dim']
    
        trainer = MUNIT_Trainer(config)
    
        try:
            state_dict = torch.load(checkpoint)
            trainer.gen_a.load_state_dict(state_dict['a'])
            trainer.gen_b.load_state_dict(state_dict['b'])
        except:
            state_dict = pytorch03_to_pytorch04(torch.load(checkpoint), opts.trainer)
            trainer.gen_a.load_state_dict(state_dict['a'])
            trainer.gen_b.load_state_dict(state_dict['b'])
    
        trainer.cuda()
        trainer.eval()
        encode = trainer.gen_a.encode if opts.a2b else trainer.gen_b.encode # encode function
        style_encode = trainer.gen_b.encode if opts.a2b else trainer.gen_a.encode # encode function
        decode = trainer.gen_b.decode if opts.a2b else trainer.gen_a.decode # decode function
        return encode, style_encode, decode
    
    encode, style_encode, decode = load_checkpoint(args.checkpoint)

    def create_image(tensor, filename, nrow=8, padding=2,
                   normalize=False, range=None, scale_each=False, pad_value=0):
        """Save a given Tensor into an image file.
    
        Args:
            tensor (Tensor or list): Image to be saved. If given a mini-batch tensor,
                saves the tensor as a grid of images by calling ``make_grid``.
            **kwargs: Other arguments are documented in ``make_grid``.
        """
        grid = vutils.make_grid(tensor, nrow=nrow, padding=padding, pad_value=pad_value,
                         normalize=normalize, range=range, scale_each=scale_each)
        ndarr = grid.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
        im = Image.fromarray(ndarr)
        return im
    
    def translate(img_input, img_style=None, style_list=None):
        with torch.no_grad():
            transform = transforms.Compose([transforms.Resize(new_size),
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            image = Variable(transform(img_input.convert('RGB')).unsqueeze(0).cuda())
            if img_style is not None:
                style_image = Variable(transform(img_style.convert('RGB')).unsqueeze(0).cuda())
                _, style = style_encode(style_image)
            elif style_list is not None:
                style = torch.Tensor(style_list).reshape([1, -1, 1, 1]).cuda()
    
            content, _ = encode(image)
    
            s = style[0].unsqueeze(0)
            outputs = decode(content, s)
            outputs = (outputs + 1) / 2.
            im = create_image(outputs, '', padding=0, normalize=True)
            return im, s.flatten()
    
    labels_loc = os.path.join(args.outdir, "labels", partition)
    images_loc = os.path.join(args.outdir, "images", partition)
    annotations_loc = os.path.join(args.outdir, "annotations")
    metadata_loc = os.path.join(args.outdir, "munit_metadata")

    # Updated annotations json
    os.makedirs(labels_loc, exist_ok=True)
    os.makedirs(images_loc, exist_ok=True)
    os.makedirs(annotations_loc, exist_ok=True)
    
    if args.blockidx == 0:    
        # Replicate images, annotations
        n_images, n_annotations = len(ann["images"]), len(ann["annotations"])

        all_images = [None] * n_images * args.nvar
        all_anns = [None] * n_annotations * args.nvar
        
        # image_id is not necessarily equal to its array index
        image_id_to_arr_idx_map = {}
        for ix, ann_img in enumerate(ann['images']):
            image_id_to_arr_idx_map[ann_img['id']] = ix + 1
        
        for i in range(args.nvar):
            for j in range(n_images):
                all_images[n_images * i + j] = copy.deepcopy(ann["images"][j])
                all_images[n_images * i + j]["id"] = n_images * i + j + 1

                bname = os.path.splitext(os.path.basename(ann["images"][j]["file_name"]))
                bname = bname[0] + "-{}".format(i) + bname[1]
                bname = os.path.abspath(os.path.join(images_loc, bname))

                all_images[n_images * i + j]["file_name"] = bname
                all_images[n_images * i + j]["coco_url"] = bname

        for i in range(args.nvar):
            for j in range(n_annotations):
                all_anns[n_annotations * i + j] = copy.deepcopy(ann["annotations"][j])
                all_anns[n_annotations * i + j]["id"] = n_annotations * i + j + len(all_images) + 1
                all_anns[n_annotations * i + j]["image_id"] = n_images * i + image_id_to_arr_idx_map[ann["annotations"][j]["image_id"]]

        ann["images"] = all_images
        ann["annotations"] = all_anns

        annotations_path = os.path.join(annotations_loc, "instances_{}.json".format(partition))
        with open(annotations_path, "w") as f:
            json.dump(ann, f)

    metadata = {}

    for sip_ix, synthetic_image_path in enumerate(synthetic_images_list[args.blockidx::args.blocksize]):
        print("Progress: {}/{}".format(sip_ix, len(synthetic_images_list[args.blockidx::args.blocksize])))
        synthetic_image = Image.open(synthetic_image_path)
        if args.in_width != -1 and args.out_width != -1:
            synthetic_image = synthetic_image.resize((args.in_width, args.in_height), Image.ANTIALIAS)
        real_images = [real_images_list[i] for i in list(np.random.permutation(len(real_images_list))[:args.nvar])]

        for style_ix, real_image_path in enumerate(real_images):
            real_image = Image.open(real_image_path)
            output_image, style_vector = translate(synthetic_image, img_style=real_image)

            metadata[(synthetic_image_path, real_image_path)] = {"real_style_vec": style_vector}

            # Save image
            bname = os.path.splitext(os.path.basename(synthetic_image_path))
            bname = bname[0] + "-" + str(style_ix) + bname[1]

            gen_image_path = os.path.join(images_loc, bname)
            output_image = output_image.resize((args.out_width, args.out_height), Image.ANTIALIAS)
            output_image.save(gen_image_path)

            # Copy labels
            if args.format == 'yolo':
                label_file = os.path.splitext(os.path.basename(synthetic_image_path))[0] + ".txt"
                styled_label_file = os.path.splitext(os.path.basename(synthetic_image_path))[0] + "-{}.txt".format(style_ix)
                shutil.copyfile(os.path.join(synthetic_labels_base_path, label_file),
                                os.path.join(labels_loc, styled_label_file))


def main():
    args = parse_args()
    for split in args.split_names:
        runner(args, split)

if __name__ == "__main__":
    main()


