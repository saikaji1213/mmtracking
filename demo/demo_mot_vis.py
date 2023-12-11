# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp
import tempfile
from argparse import ArgumentParser

import mmcv
import numpy as np

from mmtrack.apis import inference_mot, init_model
from icecream import ic
import hyperlpr3 as lpr3

import cv2
from PIL import ImageFont
from PIL import Image
from PIL import ImageDraw

from utils.color_recognition_module import color_recognition_api


def draw_plate_on_image(img, box, text, font):
    x1, y1, x2, y2 = box
    # cv2.rectangle(img, (x1, y1), (x2, y2), (139, 139, 102), 2, cv2.LINE_AA)
    # cv2.rectangle(img, (x1, y1 - 20), (x2, y1), (139, 139, 102), -1)
    data = Image.fromarray(img)
    draw = ImageDraw.Draw(data)
    draw.text((x1 + 5, y1 - 20), text, (255, 255, 255), font=font)
    res = np.asarray(data)
    return res


def main():
    parser = ArgumentParser()
    parser.add_argument('config', help='config file')
    parser.add_argument('--input', help='input video file or folder')
    parser.add_argument(
        '--output', help='output video file (mp4 format) or folder')
    parser.add_argument('--checkpoint', help='checkpoint file')
    parser.add_argument(
        '--score-thr',
        type=float,
        default=0.0,
        help='The threshold of score to filter bboxes.')
    parser.add_argument(
        '--device', default='cuda:0', help='device used for inference')
    parser.add_argument(
        '--show',
        action='store_true',
        help='whether show the results on the fly')
    parser.add_argument(
        '--backend',
        choices=['cv2', 'plt'],
        default='cv2',
        help='the backend to visualize the results')
    parser.add_argument('--fps', help='FPS of the output video')
    args = parser.parse_args()
    assert args.output or args.show
    # load images
    if osp.isdir(args.input):
        imgs = sorted(
            filter(lambda x: x.endswith(('.jpg', '.png', '.jpeg')),
                   os.listdir(args.input)),
            key=lambda x: int(x.split('.')[0]))
        IN_VIDEO = False
    else:
        imgs = mmcv.VideoReader(args.input)
        IN_VIDEO = True
    # define output
    if args.output is not None:
        if args.output.endswith('.mp4'):
            OUT_VIDEO = True
            out_dir = tempfile.TemporaryDirectory()
            out_path = out_dir.name
            _out = args.output.rsplit(os.sep, 1)
            if len(_out) > 1:
                os.makedirs(_out[0], exist_ok=True)
        else:
            OUT_VIDEO = False
            out_path = args.output
            os.makedirs(out_path, exist_ok=True)

    fps = args.fps
    if args.show or OUT_VIDEO:
        if fps is None and IN_VIDEO:
            fps = imgs.fps
        if not fps:
            raise ValueError('Please set the FPS for the output video.')
        fps = int(fps)

    # build the model from a config file and a checkpoint file
    model = init_model(args.config, args.checkpoint, device=args.device)

    # ic(len(model.CLASSES))
    # ic(model.CLASSES)

    font_ch = ImageFont.truetype("./demo/font/platech.ttf", 20, 0)
    catcher = lpr3.LicensePlateCatcher(detect_level=lpr3.DETECT_LEVEL_HIGH)
    car_track = {}  # map{id: list(license, color, start_frame, end_frame)}
    pre_frame_ids = set()
    cur_frame_ids = set()

    prog_bar = mmcv.ProgressBar(len(imgs))
    # test and show/save the images
    for i, img in enumerate(imgs):
        if isinstance(img, str):
            img = osp.join(args.input, img)
        result = inference_mot(model, img, frame_id=i)

        # dealing result
        car_refer_idxs = [68, 71, 85, 122, 186, 196, 232, 264, 272, 309, 331, 350, 444, 449]
        for idx in range(len(result['det_bboxes'])):
            if idx not in car_refer_idxs:
                result['det_bboxes'][idx] = np.zeros((0, 5), dtype='float32')
                result['track_bboxes'][idx] = np.zeros((0, 6), dtype='float32')
            else:
                # track_id不变期间识别截取track_bbox识别车牌，识别到不再识别
                pre_frame_ids = cur_frame_ids
                cur_frame_ids = set()
                for track_bbox in result['track_bboxes'][idx]:
                    if not track_bbox[-1] > args.score_thr:
                        continue
                    cur_frame_ids.add(track_bbox[0])
                    # id first appearance
                    if track_bbox[0] not in car_track:
                        bbox = track_bbox[1:-1]
                        car_image = mmcv.imcrop(img, bbox)
                        car_track_color = color_recognition_api.color_recognition(car_image)
                        car_track[track_bbox[0]] = [None, car_track_color, i, len(imgs)]
                        license_results = catcher(car_image)  # list(code, confidence, type_idx, box)
                        if license_results:
                            license_result_idx = np.argmax(np.asarray(license_results, dtype=object), axis=0)[1]
                            car_track[track_bbox[0]][0] = f"{license_results[license_result_idx][0]}"
                    else:
                        if car_track[track_bbox[0]][0] is None or "":
                            bbox = track_bbox[1:-1]
                            car_image = mmcv.imcrop(img, bbox)
                            license_results = catcher(car_image)
                            if license_results:
                                license_result_idx = np.argmax(np.asarray(license_results, dtype=object), axis=0)[1]
                                car_track[track_bbox[0]][0] = f"{license_results[license_result_idx][0]}"
                    text = car_track[track_bbox[0]][0]
                    color = car_track[track_bbox[0]][1]
                    if text is None:
                        text = 'Unknown'
                    img = draw_plate_on_image(img, track_bbox[1:-1], f"{text}|{color}", font=font_ch)
                    img = np.array(img)
                    img.flags.writeable = True
                ended_ids = [item for item in pre_frame_ids if item not in cur_frame_ids]
                for ended_id in ended_ids:
                    car_track[ended_id][3] = i

        if args.output is not None:
            if IN_VIDEO or OUT_VIDEO:
                out_file = osp.join(out_path, f'{i:06d}.jpg')
            else:
                out_file = osp.join(out_path, img.rsplit(os.sep, 1)[-1])
        else:
            out_file = None
        model.show_result(
            img,
            result,
            score_thr=args.score_thr,
            show=args.show,
            wait_time=int(1000. / fps) if fps else 0,
            out_file=out_file,
            backend=args.backend)
        prog_bar.update()

    if args.output and OUT_VIDEO:
        print(f'making the output video at {args.output} with a FPS of {fps}')
        mmcv.frames2video(out_path, args.output, fps=fps, fourcc='mp4v')
        filepath, filename = os.path.split(args.output)
        name, suffix = os.path.splitext(filename)
        absp = os.path.abspath(f"{filepath}/{name}")
        if not os.path.exists(absp):
            os.makedirs(absp)
        for track_id, feature in car_track.items():
            track_output = f"{filepath}/{name}/{track_id}-{feature[0]}-{feature[1]}-{feature[2]}-{feature[3]}{suffix}"
            print(f'making the output video at {track_output} with a FPS of {fps}')
            mmcv.frames2video(out_path, track_output, fps=fps, fourcc='mp4v', start=feature[2], end=feature[3])
        out_dir.cleanup()

    ic(car_track)


if __name__ == '__main__':
    main()
