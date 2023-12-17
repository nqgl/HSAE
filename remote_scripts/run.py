from instance import *
import argparse

parser = argparse.ArgumentParser(description='Run a command on a remote machine.')
parser.add_argument('--id', type=int, help='Name of remote machine')
parser.add_argument('--setup', action='store_true', help='Setup remote machine')
parser.add_argument('--dont-sync', action='store_true', help='Dont sync code')
parser.add_argument('--no-cd', action='store_true', help='Dont cd to ~/modified-SAE')
parser.add_argument('--re-on-save', action='store_true', help='Re-run on save')
parser.add_argument('command', type=str, help='Command to run on remote machine', nargs='+')

args = parser.parse_args()

if args.id is None:
    args.id = 0
inst = Instance.s_[args.id]
if args.setup:
    inst.setup()
if not args.dont_sync:
    inst.sync_code()

command = " ".join(args.command)
if not args.no_cd:
    cmd = f"cd ~/modified-SAE; {command}"
else:
    cmd = command
if not args.re_on_save:
    inst.run(cmd)
else:
    while True:
        try:
            inst.run(cmd)
        except:
            print("Error running command or interrupted")
        print("Waiting for change...")
        subprocess.run("inotifywait -e modify .", shell=True)
        print("Change detected!")
        if not args.dont_sync:
            inst.sync_code()

inst.close()