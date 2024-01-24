# sshcmd="ssh -p 47174 root@66.23.193.37 -L 8080:localhost:8080"

# git add *; git commit -m "push for remote access"; git push
# ${sshcmd} "cd ~/modified-SAE/ae_on_heads_w_keith; git pull; python3 train_sae.py"



instances(){
    vastai show instances -q
}



do_cmd_on_instances(){
    while read line; do
        vastai execute $line "$1"
    done
    
}

exec_all_instances(){
    instances | do_cmd_on_instances "$@"
}

return_nonzero_all_instances(){
    instances | while read line; do
        o=$(vastai execute $line "$@")
        if [ ! -z $o ]; then
            echo $o
            echo $line
        fi
    done
}

get_model_inst_loc(){
    model_name=$1
    cmd="ls /root/workspace"
    instances | while read line; do
        o=$(vastai execute "${line}" 'ls root/workspace' | g $model_name)
        if [ ! -z $o ]; then
            echo $o
            echo $line
        fi
    done
}


get_model(){
    res=$(get_model_inst_loc $1)
    # res=a
    echo ${res}
    echo "..."
    echo ${res} | tail -n 3 | {read cfg; read model; read inst}
    echo "cfg ${cfg}"
    echo "model ${model}"
    echo "inst ${inst}"
    echo "Is this correct? (y/n)"
    read yn
    if [ $yn = "y" ]; then

        pathcfg="~/workspace/${cfg}"
        pathmodel="~/workspace/${model}"
        inst_name="$(vastai scp-url ${inst})"
        # inst_name="scp://root@ssh4.vast.ai:36807"
        scp_url=${inst_name#*//}
        scp_url=${scp_url%:*}

        dest="./models-from-remote/"
        # like such as root@ssh4.vast.ai:36807:/root/workspace/93_fast-jazz-451.pt:/root/workspace/93_fast-jazz-451.pt ./model-from-remote/
        # with the port in the middle
        # we want to set the scp url to be the first part
        # and the destination to be the second part
        # port=${scp_url%*:}
        # scp_url=${inst_name%:*}
        port=${inst_name#*:}
        port=${port#*:}
        echo "scp_url ${scp_url}"
        echo "dest ${dest}"
        echo "port ${port}"
        echo scp -P ${port} ${scp_url}:${pathcfg} ${dest}
        scp -P ${port} ${scp_url}:${pathcfg} ${dest}
        echo scp -P ${port} ${scp_url}:${pathmodel} ${dest}
        scp -P ${port} ${scp_url}:${pathmodel} ${dest}
        echo done
    else
        echo "Try again"
    fi
}

inst_to_sshurl(){
    while read inst; do
        inst_name="$(vastai ssh-url ${inst})"
        if [ $1 = "-q" ]; then
            echo ${inst_name}
        else
            echo ${inst}"->"${inst_name}
        fi
    done
    
}

download_scp_link(){
    # take a file url like
    # scp://root@ssh4.vast.ai:36807:/root/workspace/93_fast-jazz-451.pt
    # and download it to the current directory
    scp_url=$1
    scp_url=${scp_url#*//}
    echo $scp_url
    
}