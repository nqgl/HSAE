from instance import *
import argparse
import time

parser = argparse.ArgumentParser(description="Run a command on a remote machine.")
parser.add_argument("--id", type=int, help="Name of remote machine")
parser.add_argument("--setup", action="store_true", help="Setup remote machine")
parser.add_argument("--dont-sync", action="store_true", help="Dont sync code")
parser.add_argument("--no-cd", action="store_true", help="Dont cd to ~/modified-SAE")
parser.add_argument("--re-on-save", action="store_true", help="Re-run on save")
parser.add_argument("--local", action="store_true", help="Run locally")
parser.add_argument(
    "--pkill",
    action="store_true",
    help="Pkill root after every command to prevent machine getting stuck.",
)
parser.add_argument(
    "--timeout", type=int, help="run command with a timeout (with --kill!)"
)
parser.add_argument(
    "command", type=str, help="Command to run on remote machine", nargs="+"
)
parser.add_argument("--copymodel", type=str, help="select a model to copy to the remote")
parser.add_argument("--tmux", action="store_true", help="get ssh for remote")
parser.add_argument(
    "--export",
    action="store_true",
    help="Export PYTHONPATH to include ~/",
)




#export PYTHONPATH=~/:$PYTHONPATH; pip install transformer_lens; cd modified-SAE; python3 train_hsae_sae0.py
#mkdir nqgl; mv modified-SAE sae; mv sae nqgl/sae; ln -s nqgl/sae; mv sae modified-SAE
args = parser.parse_args()



if args.id is None:
    args.id = 0
    if args.tmux:
        # print(args.command)
        args.id = int(args.command[0])
inst = Instance.s_[args.id]

if args.copymodel:
    inst.connect()
    inst.copy_model(args.copymodel)

if args.tmux:
    print(inst.ssh_str())
    exit()


command = " ".join(args.command)
if args.timeout:
    command = f"timeout {args.timeout} {command}"

if not args.no_cd and not args.local:
    cmd = f"cd ~/modified-SAE; {command}"
else:
    cmd = command

if args.export:
    cmd = f"export PYTHONPATH=~/:$PYTHONPATH; {cmd}"

if args.local:
    if args.pkill:
        raise Exception("Can't pkill root locally.")
    cmd = command

    if not args.re_on_save:
        subprocess.run(cmd, shell=True)
    else:
        while True:
            try:
                subprocess.run(cmd, shell=True)
            except:
                print("Error running command or interrupted")
            print("\n\n\nWaiting for change...")
            subprocess.run(
                "inotifywait -r --exclude __pycache__ -e modify .", shell=True
            )
            print("Change detected!")
    exit()

if args.setup:
    inst.setup()
if not args.dont_sync:
    inst.sync_code()

try:
    if not args.re_on_save:
        inst.run(cmd)
    else:
        t0 = time.time()
        n_fail = 0
        fail = False
        while True:
            if not args.dont_sync:
                try:
                    inst.sync_code()
                except:
                    print("Error syncing code")
                    time.sleep(1)
                    continue
            t1 = time.time()
            try:
                inst.run(cmd)
            except:
                print("Error running command or interrupted")
            finally:
                fail = time.time() - t1 < 20
                n_fail += fail
                if args.pkill:
                    try:
                        inst.run("pkill -u root")
                    except:
                        # if t0 + 16 > time.time():
                        #     time.sleep(5)
                        #     continue
                        if fail:
                            print("exec failed. Retrying after progressive sleep.")
                            time.sleep(n_fail * 5)
                            inst.close()
                            continue

                inst.close()
            print("Waiting for change...")
            subprocess.run(
                "inotifywait -r --exclude __pycache__ -e modify .", shell=True
            )
            print("Change detected!")
            time.sleep(1)
            t0 = time.time()
        n_fail = 0
except:
    pass
finally:
    if args.pkill:
        try:
            inst.run("pkill -u root")
        except:
            inst.close()
            exit()
        else:
            print("pkill failed")
    inst.close()
