git add *; git commit -m "push for remote access"; git push
ssh -p 47174 root@66.23.193.37 -L 8080:localhost:8080 "cd ~/modified-SAE/ae_on_heads_w_keith; git pull; python3 train_sae.py"
