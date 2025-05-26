import argparse
import json
import subprocess

with open("slurm/template.slurm", "r") as file:
    slurm_script = file.read()


def get_args_parser():
    parser = argparse.ArgumentParser('Launch experients')

    parser.add_argument('--config', type=str, default="configs/models/diffusiondet_dota.json")
    parser.add_argument('--dataset-names', nargs='+', default=["detection-datasets/coco"])
    parser.add_argument('--output-dir', type=str, default="diffusiondet")

    parser.add_argument('--exec-type', type=str, default="python", choices=["slurm", "python"])
    parser.add_argument('--slurm-template', type=str, default="slurm/default.slurm")

    return parser


def build_cmd(config):
    cmd = ""
    for key, value in config.items():
        if key not in ['freeze_modules', 'freeze_at'] or value != '':
            cmd += f" --{key} "
            cmd += str(value)
    return cmd


def submit_job(cmd, exec_type, **kwargs):
    if exec_type == "slurm":
        formated_script = slurm_script.format(job_name=kwargs['dataset'], command=cmd)
        with open('launchers/automatic_launcher.slurm', 'w') as f:
            f.write(formated_script)
        print('job_name', kwargs['dataset'])
        return subprocess.call(['sbatch', 'launchers/automatic_launcher.slurm'])
    elif exec_type == "python":
        cmd = f"python run_object_detection.py{cmd}"
        return subprocess.run(cmd, shell=True)
    return None


def main(args):
    with open(args.config) as f:
        config = json.load(f)

    for dataset_name in args.dataset_names:
        output_dir = f"runs/{args.output_dir}/{dataset_name.rstrip('/').split('/')[-1]}/classic_detection"

        config["dataset_name"] = dataset_name
        config["output_dir"] = output_dir
        config["logging_strategy"] = "epoch"
        config["eval_strategy"] = "epoch"
        config["save_strategy"] = "epoch"

        cmd = build_cmd(config)
        result = submit_job(cmd, exec_type=args.exec_type, dataset=dataset_name)
        if result != 0:
            print(f"Error running command: python run_object_detection.py{cmd}")
            return


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()

    main(args)
