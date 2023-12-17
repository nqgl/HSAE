# import fabric
from fabric import Connection
from patchwork.transfers import rsync
import subprocess
import paramiko
import os
from typing import Dict, List

class Instance:
    instances = {}
    s_ :List["Instance"] = []
    def __init__(self, id, url, name=None):
        self.name = str(id) if name is None else name
        self.url = url
        self.id = id
        self.c = None

    def connect(self):
        self.c = Connection(
            self.url, 
            connect_kwargs={
                "timeout" : 10,
            }
        )
    
    def run(self, cmd):
        print("\n### " + cmd)
        if self.c is None:
            self.connect()
        return self.c.run(cmd)
    

    def setup(self):
        self.run("pip install transformer-lens")
        self.run("pip install pytest")
        self.run("pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.1.0+cu121.html")
        self.sync_code()
        self.run("mkdir -p ~/workspace/data")
        self.run("mkdir -p /workspace/data")


    def label(self):
        if str(self.id) == self.name:
            return self.name
        else:
            return f"{self.name} ({self.id})"
        
    def sync_code(self):
        if self.c is None:
            self.connect()
        exclude = open(".gitignore").read().split("\n")
        exclude += [".git"]
        rsync(
            c = self.c,
            source = ".",
            target = "~/modified-SAE",
            exclude = exclude
        )

    def close(self):
        self.c.close()
        self.c = None

def get_instances():
    commands = ". ./remote_scripts/remote_run.sh; instances | inst_to_sshurl"
    r = subprocess.run(commands, shell=True, capture_output=True)
    s = r.stdout.decode("utf-8")
    insts = s.split("\n")
    print(insts)
    for insturl in insts:
        if insturl:
            id, url = insturl.split("->")
            url = url.split("ssh://")[1]
            if not id in Instance.instances:
                inst = Instance(id, url)
                Instance.instances[id] = inst
                Instance.s_.append(inst)

def AuthConnection(url):
    """
    return a connection object with private key using prompt-for-passphrase to unlock encrypted private key
    """
    c = Connection(
        url, 
        connect_kwargs={
            "timeout" : 10,
        }
    )
    return c

def run_on_instances(cmd, insts=None):
    print("run_on_instances")
    if insts is None:
        get_instances()
        print("set")
        insts = [Instance.instances[i] for i in Instance.instances]
    
    for inst in insts:
        print(f"Running {cmd} on {inst.name}")
        inst.run(cmd)

get_instances()
# inst = Instance.s_[0]
# inst.setup()
# inst.connect()
# inst.sync_code()
# # inst.run("ls transferred3")
# inst.run("ls")
# inst.run("ls; cd modified-SAE; ls")
# 
