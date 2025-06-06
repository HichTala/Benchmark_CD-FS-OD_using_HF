import argparse
import json
import os.path
import subprocess


def get_args_parser():
    parser = argparse.ArgumentParser('Launch experients')

    parser.add_argument('--config', type=str, default="configs/models/diffusiondet_dota.json")
    parser.add_argument('--dataset-names', nargs='+', default=["detection-datasets/coco"])
    parser.add_argument('--seed', nargs='+', default=["1338"])
    parser.add_argument('--shots', nargs='+', default=["10"])
    parser.add_argument('--output-dir', type=str, default="diffusiondet")

    parser.add_argument('--freeze-mode', type=bool, default=False)
    parser.add_argument('--freeze-modules', type=str)
    parser.add_argument('--freeze-at', type=str)

    parser.add_argument('--use-lora', type=bool, default=False)
    parser.add_argument('--over-lora', type=bool, default=False)
    parser.add_argument('--lora-ranks', nargs='+', default=["8"])

    parser.add_argument('--exec-type', type=str, default="python", choices=["slurm", "python"])
    parser.add_argument('--slurm-template', type=str, default="configs/slurm/default.slurm")

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
        with open(kwargs['slurm_template'], "r") as file:
            slurm_template = file.read()
        slurm_script = slurm_template.format(job_name=kwargs['dataset'], command=cmd)
        with open('launchers/automatic_launcher.slurm', 'w') as f:
            f.write(slurm_script)
        print('job_name', kwargs['shot'] + 'nl' + kwargs['seed'])
        return subprocess.call(['sbatch', 'launchers/automatic_launcher.slurm'])
    elif exec_type == "python":
        cmd = f"python run_object_detection.py{cmd}"
        return subprocess.run(cmd, shell=True)
    return None


def main(args):
    with open(args.config) as f:
        config = json.load(f)

    if not args.freeze_modules:
        args.freeze_modules = ['', 'backbone', 'backbone', 'bias', 'norm']
    if not args.freeze_at:
        args.freeze_at = ['', '0', 'half', '', '']

    for dataset_name in args.dataset_names:
        for shot in args.shots:
            for seed in args.seed:
                if args.freeze_mode:
                    for freeze_modules, freeze_at in zip(args.freeze_modules, args.freeze_at):
                        output_dir = f"runs/{args.output_dir}/{dataset_name.rstrip('/').split('/')[-1]}/{shot}/seed_{seed}/"

                        if len(freeze_modules) == 0:
                            output_dir += "full_finetuning"
                        output_dir += f"{freeze_modules}-{freeze_at}" if freeze_at != '0' else f"{freeze_modules}-full"
                        if output_dir[-1] == '-':
                            output_dir = output_dir[:-1]

                        config['freeze_modules'] = freeze_modules
                        config['freeze_at'] = freeze_at

                        config["dataset_name"] = dataset_name
                        config["seed"] = seed
                        config["shots"] = shot
                        config["output_dir"] = output_dir

                        cmd = build_cmd(config)
                        result = submit_job(cmd, exec_type=args.exec_type)
                        # if result.returncode != 0:
                        #     print(f"Error running command: python run_object_detection.py{cmd}")
                        #     return
                else:
                    output_dir = f"runs/{args.output_dir}/{dataset_name.rstrip('/').split('/')[-1]}/{shot}/nolora/{seed}"

                    logging_steps = {'50': 970, '10': 100, '5': 50, '1': 10}

                    config["dataset_name"] = dataset_name
                    config["seed"] = seed
                    config["shots"] = shot
                    config["output_dir"] = output_dir
                    config["eval_steps"] = logging_steps[shot]
                    config["save_steps"] = logging_steps[shot]

                    if args.use_lora:
                        config["use_lora"] = True
                        for rank in args.lora_ranks:
                            config["lora_rank"] = rank
                            config[
                                "output_dir"] = f"runs/{args.output_dir}/{dataset_name.rstrip('/').split('/')[-1]}/{shot}/lora/{seed}/{rank}"

                            if args.over_lora:
                                if os.path.isfile(
                                        f"runs/{args.output_dir}/{dataset_name.rstrip('/').split('/')[-1]}/{shot}/nolora/{seed}/trainer_state.json"):
                                    with open(
                                            f"runs/{args.output_dir}/{dataset_name.rstrip('/').split('/')[-1]}/{shot}/nolora/{seed}/trainer_state.json") as f:
                                        trainer_state = json.load(f)
                                    config['model_name_or_path'] = trainer_state['best_model_checkpoint']
                                    config[
                                        "output_dir"] = f"runs/{args.output_dir}/{dataset_name.rstrip('/').split('/')[-1]}/{shot}/overlora/{seed}/{rank}"

                            cmd = build_cmd(config)
                            result = submit_job(cmd, exec_type=args.exec_type, seed=seed, shot=shot, dataset=dataset_name, slurm_template=args.slurm_template)
                            if result != 0:
                                print(f"Error running command: python run_object_detection.py{cmd}")
                                return
                    else:
                        cmd = build_cmd(config)
                        result = submit_job(cmd, exec_type=args.exec_type, seed=seed, shot=shot, dataset=dataset_name, slurm_template=args.slurm_template)
                        if result != 0:
                            print(f"Error running command: python run_object_detection.py{cmd}")
                            return


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()

    main(args)
