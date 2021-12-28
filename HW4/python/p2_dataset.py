import os
from PIL import Image

from torch.utils.data import Dataset
from torchvision import transforms

dict = {
    "Couch":        0,
    "Helmet":       1,
    "Refrigerator": 2,
    "Alarm_Clock":  3,
    "Bike":         4,
    "Bottle":       5,
    "Calculator":   6,
    "Chair":        7,
    "Mouse":        8,
    "Monitor":      9,
    "Table":        10,
    "Pen":          11,
    "Pencil":       12,
    "Flowers":      13,
    "Shelf":        14,
    "Laptop":       15,
    "Speaker":      16,
    "Sneakers":     17,
    "Printer":      18,
    "Calendar":     19,
    "Bed":          20,
    "Knives":       21,
    "Backpack":     22,
    "Paper_Clip":   23,
    "Candles":      24,
    "Soda":         25,
    "Clipboards":   26,
    "Fork":         27,
    "Exit_Sign":    28,
    "Lamp_Shade":   29,
    "Trash_Can":    30,
    "Computer":     31,
    "Scissors":     32,
    "Webcam":       33,
    "Sink":         34,
    "Postit_Notes": 35,
    "Glasses":      36,
    "File_Cabinet": 37,
    "Radio":        38,
    "Bucket":       39,
    "Drill":        40,
    "Desk_Lamp":    41,
    "Toys":         42,
    "Keyboard":     43,
    "Notebook":     44,
    "Ruler":        45,
    "ToothBrush":   46,
    "Mop":          47,
    "Flipflops":    48,
    "Oven":         49,
    "TV":           50,
    "Eraser":       51,
    "Telephone":    52,
    "Kettle":       53,
    "Curtains":     54,
    "Mug":          55,
    "Fan":          56,
    "Push_Pin":     57,
    "Batteries":    58,
    "Pan":          59,
    "Marker":       60,
    "Spoon":        61,
    "Screwdriver":  62,
    "Hammer":       63,
    "Folder":       64
    }

class DATA(Dataset):

    def __init__(self, csv_path, data_path):
        csv_path = os.path.join(csv_path)
        lines = [x.strip() for x in open(csv_path, 'r').readlines()][1: ]

        self.file = []

        for l in lines:
            name = l.split(',')[1]
            label = dict[name.split('0')[0]]
            path = os.path.join(data_path, name)
            self.file.append((path, label))


        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.file)

    def __getitem__(self, i):
        path, label = self.file[i]
        image = self.transform(Image.open(path).convert('RGB'))
        return image, label