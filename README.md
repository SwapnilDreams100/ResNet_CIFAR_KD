# ResNet_CIFAR_KD

## Prerequisites
- Python 3.6+
- PyTorch 1.0+

## Teacher models:
Densenet
DLA

## Training
```
# First train the teacher with: 
python train_teacher.py --teacher "densenet"
python train_teacher.py --teacher "dla"


# You can train the student with: 
python train_student.py --teacher "densenet"
python train_student.py --teacher "dla"
```
# The entire structure of our Resnets architecture
![Neutron 1](https://user-images.githubusercontent.com/60310113/160039430-3bbbfb04-5efd-40d0-a340-13022651a787.png)
![Neutron 2](https://user-images.githubusercontent.com/60310113/160039429-1a365f6e-1b58-4079-8a57-43985ee053d1.png)
![Neutron 3](https://user-images.githubusercontent.com/60310113/160039427-cfb7ee71-9448-4735-a0b1-d7b2666e88e9.png)
![Neutron 4](https://user-images.githubusercontent.com/60310113/160039426-29d3c5cd-3f50-4911-9bee-1f9e3a4c8922.png)
![Neutron 5](https://user-images.githubusercontent.com/60310113/160039432-a57d46c8-0c38-4a1d-af03-6041f9596da7.png)
![Neutron 6](https://user-images.githubusercontent.com/60310113/160039431-f5eb05f0-bc74-433e-b5b2-010b2842b4c7.png)
