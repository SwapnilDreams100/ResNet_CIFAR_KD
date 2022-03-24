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
