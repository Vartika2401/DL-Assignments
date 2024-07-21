# Replace changerollno with your rollnumber as mentioned in Assignment Guidelines
import argparse
from Pipeline import *
# from Pipeline.2021571 import *

# print("Imported all modules successfully")
# P = argparse.ArgumentParser()
# P.add_argument("gpu", type=str)
# A = P.parse_args()
A = 'F'

if __name__ == "__main__":
    
    # imageDataset = [
        # ImageDataset(split="train"),
        # ImageDataset(split="val"),
        # ImageDataset(split="test")

    # ]
    
    audioDataset = [
        AudioDataset(split="train"),
    #     AudioDataset(split="val"),
    #     AudioDataset(split="test")
    ]


    Architectures = [
        # Resnet_Q1(ResidualBlock, [2, 2, 2, 2]),
        # VGG_Q2(),
        Inception_Q3(),
        # CustomNetwork_Q4()
    ]
    
    
    for network in Architectures:
        print(network.__class__.__name__)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            params=network.parameters(),
            lr=LEARNING_RATE
        )
        
        for dataset in audioDataset  :        # + audioDataset:
            print("dataset in main.py: ",dataset)
            print(len(dataset))
            print(dataset.datasplit)
            print("here",dataset.datasplit)
            if dataset.datasplit == "train":
                print(
                    "Training {} Architecture on {} split of {}".format(
                        network.__class__.__name__,
                        dataset.datasplit,
                        dataset.__class__.__name__
                    )
                )
                network.train()
                train_dataloader = DataLoader(
                    dataset=dataset,
                    batch_size=BATCH_SIZE,
                    shuffle=True,
                    num_workers=2,
                    drop_last=True
                )
                
                trainer(
                    # gpu=A.gpu,
                    gpu=A,
                    dataloader=train_dataloader,
                    network=network,
                    criterion=criterion,
                    optimizer=optimizer
                )
            
            elif dataset.datasplit == "val":
                print(
                    "Validating {} Architecture on {} split of {}".format(
                        network.__class__.__name__,
                        dataset.datasplit,
                        dataset.__class__.__name__
                    )
                )
                network.train()
                val_dataloader = DataLoader(
                    dataset=dataset,
                    batch_size=BATCH_SIZE,
                    shuffle=True,
                    num_workers=2,
                    drop_last=True
                )
                
                validator(
                    gpu=A,
                    dataloader=val_dataloader,
                    network=network,
                    criterion=criterion,
                    optimizer=optimizer
                )
                
            elif dataset.datasplit == "test":
                print(
                    "Testing {} Architecture on {} split of {}".format(
                        network.__class__.__name__,
                        dataset.datasplit,
                        dataset.__class__.__name__
                    )
                )
                network.eval()
                test_dataloader = DataLoader(
                    dataset=dataset,
                    batch_size=BATCH_SIZE,
                    shuffle=True,
                    num_workers=2,
                    drop_last=True
                )
