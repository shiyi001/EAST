import torch
from torch.autograd import Variable
import os 
from torch import nn
from torch.optim import lr_scheduler
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader
from torchvision import transforms
from model import East
from loss import *
from data_utils import custom_dset, collate_fn
import time
from tensorboardX import SummaryWriter
import argparse

# writer = SummaryWriter()

parser = argparse.ArgumentParser(description='PyTorch EAST Training')
parser.add_argument('--data-img', required=True,
                    help='path to img dataset')
parser.add_argument('--data-txt', required=True,
                    help='path to label dataset')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=1500, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=16, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--save-freq', default=500, type=int,
                    metavar='N', help='save frequency (default: 500)')
parser.add_argument('--pretrained', dest='pretrained', default='',
                    help='use pre-trained model')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')

def train(epochs, model, trainloader, crit, optimizer,
         scheduler, save_step, weight_decay):

    for e in range(epochs):
        print('*'* 10)
        print('Epoch {} / {}'.format(e + 1, epochs))
        model.train()
        start = time.time()
        loss = 0.0
        total = 0.0
        for i, (img, score_map, geo_map, training_mask) in enumerate(trainloader):
            scheduler.step()
            optimizer.zero_grad()
    
            img = Variable(img.cuda())
            score_map = Variable(score_map.cuda())
            geo_map = Variable(geo_map.cuda())
            training_mask = Variable(training_mask.cuda())
            f_score, f_geometry = model(img)
            loss1 = crit(score_map, f_score, geo_map, f_geometry, training_mask)
            
            loss += loss1.data[0]
            
            loss1.backward()
            optimizer.step()
        
        during = time.time() - start
        print("Loss : {:.6f}, Time:{:.2f} s ".format(loss/len(trainloader), during))
        print()
        # writer.add_scalar('loss', loss / len(trainloader), e)
        
        if (e + 1) % save_step == 0:
            if not os.path.exists('./checkpoints'):
                os.mkdir('./checkpoints')
            torch.save(model.state_dict(), './checkpoints/model_{}.pth'.format(e + 1))
        

def main():
    global args
    args = parser.parse_args()

    print (args.data_img)
    print (args.data_txt)
    train_data = custom_dset(args.data_img, args.data_txt)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_fn, num_workers=args.workers)

    model = East(args.pretrained)
    # model = model.cuda()
    model = torch.nn.DataParallel(model).cuda()

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            pretrained_dict = torch.load(args.resume)
            model.load_state_dict(pretrained_dict, strict=True)
            print("=> loaded checkpoint '{}'".format(args.resume))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    crit = LossFunc()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10000, 
                                    gamma=0.94)   
    
    train(epochs=args.epochs, model=model, trainloader=train_loader,
          crit=crit, optimizer=optimizer,scheduler=scheduler, 
          save_step=args.save_freq, weight_decay=args.weight_decay)

    # write.close()

if __name__ == "__main__":
    main()
    