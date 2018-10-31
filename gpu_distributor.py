# -*- coding: utf-8 -*-
from __future__ import print_function

import os
import sys
import argparse
import datetime
import shutil
import subprocess
import shlex

from multiprocessing import Process, Queue

import colorama
colorama.init()

def parse_arguments():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--gpus', help='numbers of available GPUs',
                        nargs='+', type=int, required=True)
    parser.add_argument('--cmd', help='command to be run (with {gpu} and {x} placeholders)')
    parser.add_argument('xs', help='values to be substituted into command',
                        nargs='+')
    parser.add_argument('--tmp_dir', help='create git worktree here')
    parser.add_argument('--last_clean_git', help='ignore dirty repo and use the last commit', action='store_true')
    return parser.parse_args()

def mkdirs(directory, clean=False):
    ''' make directories, optionaly cleaning them before '''
    if clean:
        try:
            shutil.rmtree(directory, ignore_errors=True)
        except Exception as e:
            pass

    if not os.path.exists(directory):
        os.makedirs(directory)

def git_dirty(repo_dir):
    """Check whether a git repository has uncommitted changes."""
    output = subprocess.check_output(["git", "status", "-uno", "--porcelain"], cwd=repo_dir)
    return output.strip() != b""

def pretty_time(ms):
    second = 1000
    minute = 1000 * 60
    hour = 1000 * 60 * 60

    hours = ms // hour
    minutes = (ms % hour) // minute
    seconds = (ms % minute) // second
    milliseconds = ms % second

    return "{}:{:02d}:{:02d}.{:03d}".format(hours, minutes, seconds, milliseconds)

def thread_worker(queue, gpu_id, result_queue, working_dir):
    while True:
        try:
            cmd, x = queue.get(block=False)
        except:
            break
        cmd = cmd.format(gpu=gpu_id, x=x)
        cmd = shlex.split(cmd)
        print('running {} on GPU {}'.format(x, gpu_id))
        start_time = datetime.datetime.now()
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=working_dir)
        stdout, stderr = proc.communicate()
        end_time = datetime.datetime.now()
        time_diff_ms = int((end_time - start_time).total_seconds() * 1000)
        print('finished {} after {}'.format(x, pretty_time(time_diff_ms)))

        if proc.returncode != 0:
            result_queue.put((False, x, time_diff_ms))
            print(stderr)
        else:
            result_queue.put((True, x, time_diff_ms))

def create_worktree(base_dir, uniq_name):
    tmp_path = os.path.join(base_dir, uniq_name)
    mkdirs(base_dir, clean=False)
    subprocess.check_call(["git", "worktree", "add", tmp_path, "-b", "experiment/{}".format(uniq_name)])
    return tmp_path

def clean_worktree(base_dir, uniq_name):
    tmp_path = os.path.join(base_dir, uniq_name)
    shutil.rmtree(tmp_path, ignore_errors=True)
    subprocess.check_call(["git", "worktree", "prune"])
    subprocess.check_call(["git", "branch", "-d", "experiment/{}".format(uniq_name)])

def main(args):
    if args.tmp_dir:
        if git_dirty("./") and not args.last_clean_git:
            raise RuntimeError("Git repo not clean!")
        uniq_name = str(datetime.datetime.now().date()) + '_' + str(datetime.datetime.now().time()).replace(':', '.').replace('.', '-')
        tmp_dir = args.tmp_dir
        working_dir = create_worktree(tmp_dir, uniq_name)
    else:
        working_dir = './'

    try:
        start_time = datetime.datetime.now()
        cmd_queue = Queue()
        max_l = 0
        for x in args.xs:
            cmd_queue.put((args.cmd, x))
            l = len(x)
            if l > max_l:
                max_l = l

        result_queue = Queue()

        workers = []
        for gpu_id in args.gpus:
            worker = Process(target=thread_worker, args=(cmd_queue, gpu_id, result_queue, working_dir))
            worker.daemon = True
            workers.append(worker)
            worker.start()

        for worker in workers:
            worker.join()

        end_time = datetime.datetime.now()
        total_time_ms = int((end_time - start_time).total_seconds() * 1000)

        print('-----------------------')
        print('cmd: {}'.format(args.cmd))
        print('FINISHED in {}'.format(pretty_time(total_time_ms)))
        print('-----------------------')
        while not result_queue.empty():
            status, x, time = result_queue.get()
            if status:
                status = colorama.Fore.GREEN + 'OK' + colorama.Fore.RESET
            else:
                status = colorama.Fore.RED + 'FAIL' + colorama.Fore.RESET

            formatting_string = '{:<'+str(max_l)+'} {:<14} in {}'
            print(formatting_string.format(x, status, pretty_time(time)))
    except:
        print(colorama.Fore.RED + 'whoopsie, something went wrong' + colorama.Fore.RESET)
    finally:
        if args.tmp_dir:
            clean_worktree(tmp_dir, uniq_name)

    return 0

if __name__ == '__main__':
    args = parse_arguments()
    sys.exit(main(args))
