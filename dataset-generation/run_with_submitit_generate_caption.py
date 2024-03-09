# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""
A script to run multinode training with submitit.
"""
import argparse
import os
import uuid
from pathlib import Path

import submitit_automatic_label_ram_save_json as main_func

import submitit
import copy

def parse_args():
    main_parser = main_func.get_args_parser()
    parser = argparse.ArgumentParser("Submitit for Train Data", parents=[main_parser])
    parser.add_argument("--ngpus", default=1, type=int, help="Number of gpus to request on each node")
    parser.add_argument("--nodes", default=1, type=int, help="Number of nodes to request")
    parser.add_argument("--timeout", default=2800, type=int, help="Duration of the job")
    parser.add_argument("--job_dir", default="", type=str, help="Job dir. Leave empty for automatic.")

    parser.add_argument("--partition", default="learnfair", type=str, help="Partition where to submit")
    parser.add_argument('--comment', default="", type=str,
                        help='Comment to pass to scheduler, e.g. priority message')
    return parser.parse_args()


def get_shared_folder():
    user = os.getenv("USER")
    # the shared folder should be the folder that is visible by all machines / nodes.
    if Path("/SHARED-FOLDER/").is_dir():
        p = Path('/SHARED-FOLDER/{}/Grounded-Segment-Anything/submitit'.format(user))
        p.mkdir(exist_ok=True)
        return p
    raise RuntimeError("No shared folder available")


def get_init_file():
    # Init file must not exist, but it's parent dir must exist.
    os.makedirs(str(get_shared_folder()), exist_ok=True)
    init_file = get_shared_folder() / "{}_init".format(uuid.uuid4().hex)
    if init_file.exists():
        os.remove(str(init_file))
    return init_file


class Trainer(object):
    def __init__(self, args):
        self.args = args

    def __call__(self):
        # import submitit_automatic_label_ram_save_json as main

        self._setup_gpu_args()
        main_func.main(self.args)

    def checkpoint(self):
        import os
        import submitit

        self.args.dist_url = get_init_file().as_uri()
        checkpoint_file = os.path.join(self.args.slurm_output_dir, "checkpoint.pth")
        if os.path.exists(checkpoint_file):
            self.args.resume = checkpoint_file
        print("Requeuing ", self.args)
        empty_trainer = type(self)(self.args)
        return submitit.helpers.DelayedSubmission(empty_trainer)

    def _setup_gpu_args(self):
        import submitit
        from pathlib import Path

        job_env = submitit.JobEnvironment()
        self.args.slurm_output_dir = Path(str(self.args.slurm_output_dir).replace("%j", str(job_env.job_id)))
        self.args.gpu = job_env.local_rank
        self.args.rank = job_env.global_rank
        self.args.world_size = job_env.num_tasks
        print("Process group: {} tasks, rank: {}".format(job_env.num_tasks, job_env.global_rank))


def main():
    args = parse_args()
    if args.job_dir == "":
        args.job_dir = get_shared_folder() / "%j"

    # Note that the folder will depend on the job_id, to easily track experiments
    executor = submitit.AutoExecutor(folder=args.job_dir, slurm_max_num_timeout=30)

    num_gpus_per_node = args.ngpus
    nodes = args.nodes
    timeout_min = args.timeout

    partition = args.partition
    kwargs = {}
    # if args.use_volta32:
    #     kwargs['slurm_constraint'] = 'volta32gb'
    if args.comment:
        kwargs['slurm_comment'] = args.comment

    executor.update_parameters(
        # mem_gb=40 * num_gpus_per_node,
        gpus_per_node=num_gpus_per_node,
        tasks_per_node=num_gpus_per_node,  # one task per GPU
        cpus_per_task=12,
        nodes=nodes,
        timeout_min=timeout_min,  # max is 60 * 72
        # Below are cluster dependent parameters
        slurm_partition=partition,
        slurm_signal_delay_s=120,
        slurm_account='muvigen',
        slurm_qos='muvigen',
        **kwargs
    )

    executor.update_parameters(name="train-data")

    # Since it is often necessary to submit over 100 jobs simutanously, 
    # using an array to submit these jobs is a more efficient way.
    args.dist_url = get_init_file().as_uri()
    args.slurm_output_dir = args.job_dir
    print(args.slurm_output_dir)
    # list_folders = list(range(0, 250))
    list_folders = list(range(0, args.num_jobs))
    jobs = []
    args_list = []
    for folder_index in list_folders:
        args_copy = copy.deepcopy(args)
        args_copy.job_index = folder_index
        args_list.append(args_copy)

    with executor.batch():
        for args in args_list:
            trainer = Trainer(args)
            job = executor.submit(trainer)
            jobs.append(job)
    for job in jobs:
        print("Submitted job_id:", job.job_id)


if __name__ == "__main__":
    main()