import os
import random
import shutil
import labelme
import PIL
import io
import subprocess
import json
import torchvision
import torch

def preprocess(
        raw_video_basedir,
        frame_output_basedir,
        mobilenet_train_data_output_basedir,
        train_base_dir,
        merges=[],
        expects=[]
):
    videos = []
    # probing videos frame rate etc.
    for raw_file in os.listdir(raw_video_basedir):
        raw_file_abs = os.sep.join([raw_video_basedir, raw_file])
        if (
                os.path.isfile(raw_file_abs)
                and not raw_file.startswith(".")
                and (raw_file.lower().endswith(".mp4") or raw_file.lower().endswith(".mov"))
                and raw_file not in expects
        ):
            ffprobe = [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-count_frames",
                "-show_entries",
                "stream=nb_read_frames,avg_frame_rate",
                "-of",
                "json",
                raw_file_abs,
            ]
            print(" ".join(ffprobe))
            process = subprocess.Popen(
                ffprobe, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            out, err = process.communicate()
            if err:
                print(err, process.returncode)
            video_info = json.loads(out)["streams"][0]
            frames = int(video_info["nb_read_frames"])
            frame_rate = video_info["avg_frame_rate"].split("/")
            frame_rate = float(frame_rate[0]) / float(frame_rate[1])
            videos.append((raw_file_abs, frames, frame_rate))

    _, shortest, _ = sorted(videos, key=lambda x: x[1])[0]
    # shortest = 100
    videos = sorted(videos, key=lambda x: x[0])
    names = {}
    i = -1
    # handle target merge
    for raw_file_abs, _, _ in videos:
        short_name = raw_file_abs.split("/")[-1].split(".")[0]
        i += 1
        names[short_name] = i
        for merge in merges:
            if short_name in merge:
                merge_name = merge[0]
                a = names.get(merge_name, i)
                print(short_name, merge_name, a)
                names[short_name] = a
                i = a
    # extract frame
    for raw_file_abs, frames, frame_rate in videos:
        short_name = raw_file_abs.split("/")[-1].split(".")[0]
        merged_name = str(names[short_name])
        outputpath = os.sep.join([frame_output_basedir, merged_name])
        outputfile = os.sep.join([outputpath, f"{short_name}_%d.png"])
        print(frames, shortest, frame_rate)
        ffmpeg = [
            "ffmpeg",
            "-i",
            raw_file_abs,
            "-s",
            "360x640",
            "-r",
            str(shortest / frames * frame_rate),
            outputfile,
        ]
        if not os.path.exists(outputpath):
            os.makedirs(outputpath)
        print(" ".join(ffmpeg))
        process = subprocess.Popen(
            ffmpeg, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        out, err = process.communicate()
        if err:
            print(err.decode(), process.returncode)
    # split dataset
    for klass in os.listdir(frame_output_basedir):
        klass_abs = os.sep.join([frame_output_basedir, klass])
        if (
                os.path.isdir(klass_abs)
                and not klass.startswith(".")
                and (klass != "val" or klass != "train" or klass != "mobilenet")
        ):
            targets = []
            for target in os.listdir(klass_abs):
                target_abs = os.sep.join([klass_abs, target])
                if os.path.isfile(target_abs) and target.endswith(".json"):
                    target = labelfile.filename.replace(".json", ".jpg")
                    if os.path.exists(target):
                        continue
                    labelfile = labelme.LabelFile(target)
                    image = PIL.Image.open(io.BytesIO(labelfile.imageData)).convert(
                        "RGB"
                    )
                    image.save(target)
                    targets.append(target)
                if (
                        os.path.isfile(target_abs)
                        and not target.startswith(".")
                        and (
                        target.lower().endswith(".png")
                        or target.lower().endswith(".jpg")
                        or target.lower().endswith(".jpeg")
                )
                ):
                    targets.append(target_abs)
            vals = random.sample(targets, int(len(targets) * 0.3))
            trains = set(targets) - set(vals)

            val_klass_path = os.sep.join(
                [mobilenet_train_data_output_basedir, "val", klass]
            )
            if not os.path.exists(val_klass_path):
                os.makedirs(val_klass_path)

            for val_abs in vals:
                shutil.copy(
                    val_abs, os.sep.join([val_klass_path, val_abs.split(os.sep)[-1]])
                )

            train_klass_path = os.sep.join(
                [mobilenet_train_data_output_basedir, "train", klass]
            )
            if not os.path.exists(train_klass_path):
                os.makedirs(train_klass_path)

            for train_abs in trains:
                shutil.copy(
                    train_abs,
                    os.sep.join([train_klass_path, train_abs.split(os.sep)[-1]]),
                )
    # setup symlink
    for train_base_source in os.listdir(os.sep.join([train_base_dir, 'train'])):
        if int(train_base_source) not in set(names.values()):
            train_base_source_abs = os.sep.join([train_base_dir, 'train', train_base_source])
            train_base_target_abs = os.sep.join(
                [mobilenet_train_data_output_basedir, "train", train_base_source]
            )
            if not os.path.exists(train_base_target_abs):
                os.symlink(train_base_source_abs, train_base_target_abs)

    for train_base_source in os.listdir(os.sep.join([train_base_dir, 'val'])):
        if int(train_base_source) not in set(names.values()):
            train_base_source_abs = os.sep.join([train_base_dir, 'val', train_base_source])
            train_base_target_abs = os.sep.join(
                [mobilenet_train_data_output_basedir, "val", train_base_source]
            )
            if not os.path.exists(train_base_target_abs):
                os.symlink(train_base_source_abs, train_base_target_abs)
    print(names)


def postprocess(checkpoint_pth, export_file):
    model = torchvision.models.get_model("mobilenet_v2", num_classes=1000)
    checkpoint = torch.load(
        checkpoint_pth,
        map_location="cpu",
    )
    model.load_state_dict(checkpoint["model"])
    torch.onnx.export(
        model,
        torch.rand((1, 3, 224, 224)),
        export_file,
        input_names=["input"],
        output_names=["output"],
    )


def get_args_parser(add_help=True):
    import argparse

    parser = argparse.ArgumentParser(
        description="pre and post process", add_help=add_help
    )

    parser.add_argument("--raw_video_basedir", type=str)
    parser.add_argument("--frame_output_basedir", type=str)
    parser.add_argument("--mobilenet_train_data_output_basedir", type=str)
    parser.add_argument("--train_base", default="/train/imagenet1k_mini", type=str)
    parser.add_argument("--merges", default=[], action="append", nargs="+")
    parser.add_argument("--expects", default=[], nargs="*")
    parser.add_argument("--checkpoint_pth", type=str)
    parser.add_argument("--export_file", type=str)
    parser.add_argument(
        "--mode", default="pre", type=str, choices=["pre", "post"]
    )

    return parser


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    print(args)
    if args.mode == "pre":
        preprocess(
            raw_video_basedir=args.raw_video_basedir,
            frame_output_basedir=args.frame_output_basedir,
            mobilenet_train_data_output_basedir=args.mobilenet_train_data_output_basedir,
            train_base_dir=args.train_base,
            merges=args.merges,
            expects=args.expects
        )
    elif args.mode == "post":
        postprocess(checkpoint_pth=args.checkpoint_pth, export_file=args.export_file)
