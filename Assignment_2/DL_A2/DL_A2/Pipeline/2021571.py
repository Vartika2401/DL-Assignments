import gc
import math

import torch
import torchaudio
import torchvision
import torch.nn as nn
# from Pipeline import *
from torch.utils.data import Dataset, DataLoader, random_split
import librosa
from tqdm import tqdm

checkpoint_interval = 5
"""
Write Code for Downloading Image and Audio Dataset Here
"""
image_dataset_downloader = torchvision.datasets.CIFAR10(
    root="data",
    train=True,
    download=True,
    transform=torchvision.transforms.ToTensor()
)
audio_dataset_downloader = torchaudio.datasets.SPEECHCOMMANDS(root="./", download=True)

# split the image dataset into train, test and validation
image_train_dataset, image_val_dataset = torch.utils.data.random_split(
    image_dataset_downloader,
    [int(0.8 * len(image_dataset_downloader)), int(0.2 * len(image_dataset_downloader))]
)

image_test_dataset = torchvision.datasets.CIFAR10(
    root="data",
    train=False,
    download=True,
    transform=torchvision.transforms.ToTensor()
)


class ImageDataset(Dataset):
    def __init__(self, split: str = "train") -> None:
        super().__init__()
        if split not in ["train", "test", "val"]:
            raise Exception("Data split must be in [train, test, val]")

        self.datasplit = split
        # pass
        """
        Write your code here
        """
        if split == "train":
            self.dataset = image_train_dataset
        elif split == "val":
            self.dataset = image_val_dataset
        else:
            self.dataset = image_test_dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]


class AudioDataset(Dataset):
    def __init__(self, split: str = "train") -> None:
        super().__init__()
        if split not in ["train", "test", "val"]:
            raise Exception("Data split must be in [train, test, val]")
        self.mapping = {
            "backward": 0,
            "bed": 1,
            "bird": 2,
            "cat": 3,
            "dog": 4,
            "down": 5,
            "eight": 6,
            "five": 7,
            "follow": 8,
            "forward": 9,
            "four": 10,
            "go": 11,
            "happy": 12,
            "house": 13,
            "learn": 14,
            "left": 15,
            "marvin": 16,
            "nine": 17,
            "no": 18,
            "off": 19,
            "on": 20,
            "one": 21,
            "right": 22,
            "seven": 23,
            "sheila": 24,
            "six": 25,
            "stop": 26,
            "three": 27,
            "tree": 28,
            "two": 29,
            "up": 30,
            "visual": 31,
            "wow": 32,
            "yes": 33,
            "zero": 34,
        }
        self.datasplit = split
        # speechcommand version 0.02
        self.data = []
        self.labels = []
        self.transform = torchaudio.transforms.Resample(orig_freq=16000, new_freq=8000)
        if (split == "train"):
            for i in (range(55125)):  # 78750
                waveform, sample_rate, label, speaker_id, utterance_number = audio_dataset_downloader[i]
                waveform = torch.nn.functional.pad(waveform, (0, 16000 - waveform.shape[1]))
                waveform = self.transform(waveform)
                self.data.append(waveform)
                self.labels.append(self.mapping[label])
        elif (split == "val"):
            for i in (range(55125, 67892)):
                waveform, sample_rate, label, speaker_id, utterance_number = audio_dataset_downloader[i]
                waveform = torch.nn.functional.pad(waveform, (0, 16000 - waveform.shape[1]))
                waveform = self.transform(waveform)
                self.data.append(waveform)
                self.labels.append(self.mapping[label])
        else:
            for i in (range(67892, 78750)):
                waveform, sample_rate, label, speaker_id, utterance_number = audio_dataset_downloader[i]
                waveform = torch.nn.functional.pad(waveform, (0, 16000 - waveform.shape[1]))
                waveform = self.transform(waveform)
                self.data.append(waveform)
                self.labels.append(self.mapping[label])
            # print("Loading test end")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> tuple:
        return self.data[index], self.labels[index]


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels))
        self.conv1_audio = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU())
        self.conv2_audio = nn.Sequential(
            nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(out_channels))
        self.downsample = downsample
        self.relu = nn.ReLU()
        self.out_channels = out_channels

    def forward(self, x):
        if len(x.shape) == 4:
            residual = x
            out = self.conv1(x)
            out = self.conv2(out)
            if self.downsample:
                residual = self.downsample(x)
            out += residual
            out = self.relu(out)
            return out
        else:
            residual = x
            # print("x shape", x.shape)
            # print("residual shape", residual.shape)
            out = self.conv1_audio(x)
            # print("out shape after conv1", out.shape)
            out = self.conv2_audio(out)
            # if self.downsample:
            #     residual = self.downsample(x)
            # print("out shape after conv2", out.shape)
            # print("residual shape", residual.shape)
            out += residual
            out = self.relu(out)
            return out


class Resnet_Q1(nn.Module):
    def __init__(self, block=ResidualBlock, layers=[4, 4, 6, 4], num_classes=10):
        super(Resnet_Q1, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU())
        self.conv1_audio = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU())

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # print("layer[0]")
        self.layer0 = self._make_layer(block, 64, layers[0], stride=1)
        # print("layer[1]")
        self.layer1 = self._make_layer(block, 64, layers[1], stride=1)
        self.layer2 = self._make_layer(block, 64, layers[2], stride=1)
        self.layer3 = self._make_layer(block, 64, layers[3], stride=1)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(6400, num_classes)
        self.fc_audio = nn.Linear(256000, 35)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        # if stride != 1 or self.inplanes != planes:
        #     downsample = nn.Sequential(
        #         nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride),
        #         nn.BatchNorm2d(planes),
        #     )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for i in range(1, blocks):
            # print("inplanes", self.inplanes)
            # print("planes", planes)
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        if len(x.shape) == 4:
            x = self.conv1(x)
            # x = self.maxpool(x)
            x = self.layer0(x)
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
        else:
            x = self.conv1_audio(x)
            # x = self.maxpool(x)
            x = self.layer0(x)
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            # x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            x = self.fc_audio(x)

        return x


class VGG_Q2(nn.Module):
    def __init__(self,
                 *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        """
        Write your code here
        """
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=3)
        self.conv1_audio = nn.Conv1d(1, 64, kernel_size=3, stride=1, padding=3)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=3)
        self.conv2_audio = nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=3)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=3)
        self.conv3_audio = nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=3)
        self.pool_1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool_1_audio = nn.MaxPool1d(kernel_size=2, stride=2)
        p = 64
        new = 64
        new = math.ceil(new * 0.65)
        self.conv4 = nn.Conv2d(64, new, kernel_size=4, stride=1, padding=3)
        self.conv4_audio = nn.Conv1d(64, new, kernel_size=4, stride=1, padding=3)
        self.conv5 = nn.Conv2d(new, new, kernel_size=4, stride=1, padding=3)
        self.conv5_audio = nn.Conv1d(new, new, kernel_size=4, stride=1, padding=3)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool2_audio = nn.MaxPool1d(kernel_size=2, stride=2)
        p = new
        new = math.ceil(new * 0.65)
        k_size = math.ceil(4 * 1.25)
        self.conv6 = nn.Conv2d(p, new, kernel_size=k_size, stride=1, padding=3)
        self.conv6_audio = nn.Conv1d(p, new, kernel_size=k_size, stride=1, padding=3)
        self.conv7 = nn.Conv2d(new, new, kernel_size=k_size, stride=1, padding=3)
        self.conv7_audio = nn.Conv1d(new, new, kernel_size=k_size, stride=1, padding=3)
        self.conv8 = nn.Conv2d(new, new, kernel_size=k_size, stride=1, padding=3)
        self.conv8_audio = nn.Conv1d(new, new, kernel_size=k_size, stride=1, padding=3)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool3_audio = nn.MaxPool1d(kernel_size=2, stride=2)
        p = new
        new = math.ceil(new * 0.65)
        k_size = math.ceil(k_size * 1.25)
        self.conv9 = nn.Conv2d(p, new, kernel_size=k_size, stride=1, padding=3)
        self.conv9_audio = nn.Conv1d(p, new, kernel_size=k_size, stride=1, padding=3)
        self.conv10 = nn.Conv2d(new, new, kernel_size=k_size, stride=1, padding=3)
        self.conv10_audio = nn.Conv1d(new, new, kernel_size=k_size, stride=1, padding=3)
        self.conv11 = nn.Conv2d(new, new, kernel_size=k_size, stride=1, padding=3)
        self.conv11_audio = nn.Conv1d(new, new, kernel_size=k_size, stride=1, padding=3)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool4_audio = nn.MaxPool1d(kernel_size=2, stride=2)
        p = new
        new = math.ceil(new * 0.65)
        k_size = math.ceil(k_size * 1.25)
        self.conv12 = nn.Conv2d(p, new, kernel_size=k_size, stride=1, padding=3)
        self.conv12_audio = nn.Conv1d(p, new, kernel_size=k_size, stride=1, padding=3)
        self.conv13 = nn.Conv2d(new, new, kernel_size=k_size, stride=1, padding=4)
        self.conv13_audio = nn.Conv1d(new, new, kernel_size=k_size, stride=1, padding=4)
        self.conv14 = nn.Conv2d(new, new, kernel_size=k_size, stride=1, padding=4)
        self.conv14_audio = nn.Conv1d(new, new, kernel_size=k_size, stride=1, padding=4)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool5_audio = nn.MaxPool1d(kernel_size=2, stride=2)
        self.fc1_audio = nn.Linear(3250, 2300)
        self.fc1 = nn.Linear(13, 23)
        self.fc2 = nn.Linear(23, 69)
        self.fc2_audio = nn.Linear(2300, 69)
        self.fc3 = nn.Linear(69, 10)
        self.fc3_audio = nn.Linear(69, 35)

    def forward(self, x):
        if len(x.shape) == 4:
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.conv3(x)
            x = self.pool_1(x)
            x = self.conv4(x)
            x = self.conv5(x)
            x = self.pool2(x)
            x = self.conv6(x)
            x = self.conv7(x)
            x = self.conv8(x)
            x = self.pool3(x)
            x = self.conv9(x)
            x = self.conv10(x)
            x = self.conv11(x)

            x = self.pool4(x)
            x = self.conv12(x)
            x = self.conv13(x)
            x = self.conv14(x)
            x = self.pool5(x)
            x = x.view(x.size(0), -1)
            x = self.fc1(x)
            x = self.fc2(x)
            x = self.fc3(x)
            return x
        else:
            x = self.conv1_audio(x)
            x = self.conv2_audio(x)
            x = self.conv3_audio(x)
            x = self.pool_1_audio(x)
            x = self.conv4_audio(x)
            x = self.conv5_audio(x)
            x = self.pool2_audio(x)
            x = self.conv6_audio(x)
            x = self.conv7_audio(x)
            x = self.conv8_audio(x)
            x = self.pool3_audio(x)
            x = self.conv9_audio(x)
            x = self.conv10_audio(x)
            x = self.conv11_audio(x)
            x = self.pool4_audio(x)
            x = self.conv12_audio(x)
            x = self.conv13_audio(x)
            x = self.conv14_audio(x)
            x = self.pool5_audio(x)
            x = x.view(x.size(0), -1)
            x = self.fc1_audio(x)
            x = self.fc2_audio(x)
            x = self.fc3_audio(x)
            return x


class conv_block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=False):
        super(conv_block, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(int(in_channels), int(out_channels), int(kernel_size), int(stride), int(padding), bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.layer_audio = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, bias=bias),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        if len(x.shape) == 4:
            return self.layer(x)
        else:
            return self.layer_audio(x)


class Inception_Block(nn.Module):
    def __init__(self, in_channels, b1, b2, b3
                 , bias=False):
        super(Inception_Block, self).__init__()
        self.branch1 = nn.Sequential(
            conv_block(in_channels, b1, 1, 1, 0, bias=bias)
        )
        self.branch2 = nn.Sequential(
            conv_block(in_channels, b2[0], 3, 1, 1, bias=bias),
            conv_block(b2[0], b2[1], 5, 1, 2, bias=bias)
        )
        self.branch3 = nn.Sequential(
            conv_block(in_channels, b3[0], 3, 1, 1, bias=bias),
            conv_block(b3[0], b3[1], 5, 1, 2, bias=bias)
        )
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(3, 1, 1),
        )
        self.branch1_audio = nn.Sequential(
            conv_block(in_channels, b1, 1, 1, 0, bias=bias)
        )
        self.branch2_audio = nn.Sequential(
            conv_block(in_channels, b2[0], 3, 1, 1, bias=bias),
            conv_block(b2[0], b2[1], 5, 1, 2, bias=bias)
        )
        self.branch3_audio = nn.Sequential(
            conv_block(in_channels, b3[0], 3, 1, 1, bias=bias),
            conv_block(b3[0], b3[1], 5, 1, 2, bias=bias)
        )
        self.branch4_audio = nn.Sequential(
            nn.MaxPool1d(3, 1, 1),
        )

    def forward(self, x):
        #         print(self.branch1(x).shape, self.branch2(x).shape, self.branch3(x).shape, self.branch4(x).shape)
        if len(x.shape) == 4:
            return torch.cat([self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(x)], 1)
        else:
            return torch.cat([self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(x)], 1)


class Inception_Q3(nn.Module):
    def __init__(self,
                 *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        """
        Write your code here
        """

        self.incep1 = Inception_Block(3, 15, [16, 31], [16, 31])
        # self.incep1 = Inception_Block(4, 15, [16, 31], [16, 31])
        self.incep1_audio = Inception_Block(1, 6, [16, 15], [16, 20])
        self.incep2 = Inception_Block(80, 32, [32, 64], [32, 64])
        self.incep2_audio = Inception_Block(42, 6, [32, 15], [32, 20])
        self.incep3 = Inception_Block(240, 64, [64, 128], [64, 128])
        self.incep3_audio = Inception_Block(83, 6, [64, 15], [64, 20])
        self.incep4 = Inception_Block(560, 64, [64, 128], [64, 128])
        self.incep4_audio = Inception_Block(124, 6, [64, 15], [64, 20])
        self.pool = nn.MaxPool2d(2, 2)
        self.apool = nn.MaxPool1d(2, 2)

        self.fc1 = nn.Linear(880 * 4 * 4, 10)
        self.fc_audio = nn.Linear(165000, 35)
        # self.incep1_audio = Inception_Block(3, 15, [16, 31], [16, 31])
        # self.incep2_audio = Inception_Block(80, 32, [32, 64], [32, 64])
        # self.incep3_audio = Inception_Block(240, 64, [64, 128], [64, 128])
        # self.incep4_audio = Inception_Block(560, 64, [64, 128], [64, 128])
        # self.pool_audio = nn.MaxPool1d(2, 2)

    #         self.fc2 = nn.Linear(10, 10)

    def forward(self, x):
        """
        Write your code here
        """
        if len(x.shape) == 4:
            x = self.incep1(x)
            x = self.incep2(x)
            x = self.pool(x)
            x = self.incep3(x)
            x = self.pool(x)
            x = self.incep4(x)
            x = self.pool(x)
            #         print(x.shape)
            x = x.view(-1, 880 * 4 * 4)
            #         print(x.shape)
            x = self.fc1(x)

            #         x = self.fc2(x)
            return x
        else:
            x = self.incep1_audio(x)
            x = self.incep2_audio(x)
            x = self.apool(x)
            x = self.incep3_audio(x)
            x = self.apool(x)
            x = self.incep4_audio(x)
            x = self.apool(x)
            # print("before view")
            # print(x.shape)
            #         print(x.shape)
            # make sure batch size remains same
            x = x.view(x.size(0), -1)
            # print("after view")
            # print(x.shape)
            x = self.fc_audio(x)

            #         x = self.fc2(x)
            return x


class CustomNetwork_Q4(nn.Module):
    def __init__(self,
                 *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        return "NOT DONE YET"
        """
        Write your code here
        """


def save_checkpoint(model, optimizer, epoch, checkpoint_path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, checkpoint_path + '.pt')


def trainer(gpu="F",
            dataloader=None,
            network=None,
            criterion=None,
            optimizer=None):
    device = torch.device("cuda:0") if gpu == "T" else torch.device("cpu")
    checkpoint_path = network.__class__.__name__ + dataset.__class__.__name__
    # print("check_point: ", checkpoint_path)
    network = network.to(device)
    # dataloader = dataloader.to(device)
    #     print("Training on: ", device)
    #     print("Data: ", dataloader)
    # Write your code here
    for epoch in range(10):
        #         print("Epoch: ", epoch)
        #         print(len(dataloader))
        accuracy = 0
        i = 0
        for data in (dataloader):
            inputs, labels = data
            #             print(type(inputs), type(labels))

            inputs, labels = inputs.to(device), labels.to(device)
            # print(inputs.shape, labels.shape)
            optimizer.zero_grad()
            outputs = network(inputs)
            # print(outputs.shape, labels.shape)
            #             print("hehehahas")
            loss = criterion(outputs, labels)
            loss.backward()
            #             print("back hehehahs")
            optimizer.step()
            accuracy += (outputs.argmax(1) == labels).float().mean()
            del inputs, labels, outputs
            torch.cuda.empty_cache()
            gc.collect()
            i += 1
        #             print(f"Epoch: {epoch}, Batch: {i}, Loss: {loss.item()}")

        save_checkpoint(network, optimizer, epoch, checkpoint_path)
        print("Training Epoch: {}, [Loss: {}, Accuracy: {}]".format(epoch, loss.item(), accuracy / len(dataloader)))
    # Only use this print statement to print your epoch loss, accuracy
    # print("Training Epoch: {}, [Loss: {}, Accuracy: {}]".format(
    #     epoch,
    #     loss,
    #     accuracy
    # ))
    # """


def validator(gpu="F",
              dataloader=None,
              network=None,
              criterion=None,
              optimizer=None):
    device = torch.device("cuda:0") if gpu == "T" else torch.device("cpu")

    network = network.to(device)

    # Write your code here
    # load checkpoint
    checkpoint_path = network.__class__.__name__ + dataset.__class__.__name__
    checkpoint = torch.load(checkpoint_path)
    network.load_state_dict(checkpoint['model_state_dict'])
    for epoch in range(10):
        total_loss = 0
        total_accuracy = 0
        loss = 0
        for data in (dataloader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = network(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            accuracy = (outputs.argmax(1) == labels).float().mean()
            total_accuracy += accuracy
            loss.backward()
            optimizer.step()
            #         print(f"Loss: {loss.item()}, Accuracy: {accuracy}")
        print("Validation Epoch: {}, [Loss: {}, Accuracy: {}]".format(
            epoch,
            total_loss / len(dataloader),
            total_accuracy / len(dataloader)
        ))
        save_checkpoint(network, optimizer, epoch, checkpoint_path + "val")

    """
    Only use this print statement to print your epoch loss, accuracy
    print("Validation Epoch: {}, [Loss: {}, Accuracy: {}]".format(
        epoch,
        loss,
        accuracy
    ))
    """


def evaluator(gpu='F',
              dataloader=None,
              network=None,
              criterion=None,
              optimizer=None):
    # Write your code here
    checkpoint_path = [network.__class__.__name__ + dataset.__class__.__name__,
                       network.__class__.__name__ + dataset.__class__.__name__ + "val"]
    for i in range(2):
        total_loss = 0
        total_accuracy = 0
        checkpoint = torch.load(checkpoint_path[i])
        network.load_state_dict(checkpoint['model_state_dict'])
        for data in (dataloader):
            inputs, labels = data
            outputs = network(inputs)
            loss = criterion(outputs, labels)
            accuracy = (outputs.argmax(1) == labels).float().mean()
            total_loss += loss.item()
            total_accuracy += accuracy
        #         print(f"Loss: {loss.item()}, Accuracy: {accuracy}")
        # if i == 0:
            # print("CHECKPOINT AFTER TRAINING")
        # else:
            # print("CHECKPOINT AFTER VALIDATION")
        print("[Loss: {}, Accuracy: {}]".format(
            total_loss / len(dataloader),
            total_accuracy / len(dataloader)
        ))

    """
    Only use this print statement to print your loss, accuracy
    print("[Loss: {}, Accuracy: {}]".format(
        loss,
        accuracy
    ))
    """

