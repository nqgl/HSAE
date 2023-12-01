import wandb


wandb.login(key="763967cb34da114063379b7b42fec47c0be2fdb8")

excfg = {
    "ex1" : 1,
    "str_ex1" : "a string"
}

run = wandb.init(
    # Set the project where this run will be logged
    project="this-is-a-test-project",
    # Track hyperparameters and run metadata
    config=excfg,
    name="not-denim-energy?"
)

for i in range(100):
    wandb.log({"number": i})