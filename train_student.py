import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import random, os
import numpy as np
from teacher_models import *
from project1_model import *

import argparse
parser = argparse.ArgumentParser()

parser.add_argument('--teacher', type=str)

def seed_everything(seed: int):    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    
seed_everything(42)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(
    root='./', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(
    root='./', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

def test_teach(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    # Save checkpoint.
    acc = 100.*correct/total
    print(acc)

if args.teacher == 'dla':
  net = SimpleDLA()
elif args.teacher == 'densenet':
  net = DenseNet121()
else:
  print("Please enter correct teacher")

net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

checkpoint = torch.load('./checkpoint_teacher/ckpt_'+args.teacher+'_final.pt')
net.load_state_dict(checkpoint)
criterion = nn.CrossEntropyLoss()
net.eval()
test_teach(1)

# Training
def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, output_student, y_a, y_b, lam, output_teacher_batch):
    return lam * criterion(output_student, y_a, output_teacher_batch) + (1 - lam) * criterion(output_student, y_b, output_teacher_batch)

def loss_fn_kd(outputs, labels, teacher_outputs):
    """
    Compute the knowledge-distillation (KD) loss given outputs, labels.
    "Hyperparameters": temperature and alpha
    NOTE: the KL Divergence for PyTorch comparing the softmaxs of teacher
    and student expects the input tensor to be log probabilities! See Issue #2
    """
    params = {
            "alpha": 0.95,
            "temperature": 6,
    }
    alpha = params['alpha']
    T = params['temperature']
    KD_loss = nn.KLDivLoss()(F.log_softmax(outputs/T, dim=1),
                             F.softmax(teacher_outputs/T, dim=1)) * (alpha * T * T) + \
              F.cross_entropy(outputs, labels) * (1. - alpha)

    return KD_loss

def train_kd(epoch):
    print('\nEpoch: %d' % epoch)
    student.train()
    net.eval()
    
    train_loss = 0
    correct = 0
    total = 0
    losses = 0.0
    
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        inputs, targets_a, targets_b, lam = mixup_data(inputs, targets,
                                                       1.0, True)
        inputs, targets_a, targets_b = map(Variable, (inputs,
                                                      targets_a, targets_b))

        optimizer.zero_grad()
        with torch.no_grad():
            output_teacher_batch = net(inputs)
            
        outputs_student = student(inputs)
        loss = mixup_criterion(loss_fn_kd, outputs_student, targets_a, targets_b, lam, output_teacher_batch)
        losses += loss.item()
        
        loss.backward()
        optimizer.step()
    return losses/len(trainloader)

# Testing
def test_kd(epoch):
    global best_acc
    student.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = student(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    # Save checkpoint.
    acc = 100.*correct/total
    print(acc)
    if acc > best_acc:
        print('Saving..')
        
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(student.state_dict(), './checkpoint/student_best.pt')
        best_acc = acc
    if epoch == 250:
        torch.save(student.state_dict(), './checkpoint/student_final.pt')
        
    return test_loss/len(trainloader)

def adjust_learning_rate(optimizer, epoch):
    """decrease the learning rate at 100 and 150 epoch"""
    lr = 0.1
    if epoch >= 100:
        lr /= 10
    if epoch >= 150:
        lr /= 10
    if epoch >= 200:
        lr /= 10
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def count_parameters(model):
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        param = parameter.numel()
        total_params+=param
    return total_params

student = project1_model()
student.to(device)
total_params = count_parameters(student)
print(total_params)
train_losses = []
test_losses = []
if total_params < 5000000:
    if device == 'cuda':
        student = torch.nn.DataParallel(student)
        cudnn.benchmark = True

    best_acc=0
    start_epoch = 0

    optimizer = optim.SGD(student.parameters(), lr=0.1,momentum=0.9, 
                          weight_decay=5e-4)
#     scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=250)
    for epoch in range(start_epoch, start_epoch+201):
        train_loss = train_kd(epoch)
        test_loss = test_kd(epoch)
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        adjust_learning_rate(optimizer, epoch)
#         scheduler.step()
    print(best_acc)
