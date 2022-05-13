import torch
import os
import sys
from backbone.model_irse import IR_50
from backbone.model_irse_drop import IR_50_drop
import cv2
import numpy as np
import argparse
import datetime
from attack.privacy import cos_sim, FIM, DFANet_MFIM

def main(args):
    print(args)

    time1 = datetime.datetime.now()

    # initialize the surrogate model
    DEVICE = torch.device("cuda:0")
    model_root = args.pretrained
    if args.DFANet == 1:
        model = IR_50_drop([112, 112], args.droprate)
        dict_trained = torch.load(model_root)
        dict_new = model.state_dict().copy()

        new_list = list(model.state_dict().keys())
        trained_list = list(dict_trained.keys())
        print("new_state_dict size: {}  trained state_dict size: {}".format(len(new_list), len(trained_list)))

        for i in range(len(trained_list)):
            if 'input_layer.1' in trained_list[i]:
                new_name = 'input_layer.2' + trained_list[i].split('input_layer.1')[-1]
                print('new name', trained_list[i], new_name)
            elif 'input_layer.2' in trained_list[i]:
                new_name = 'input_layer.3' + trained_list[i].split('input_layer.2')[-1]
                print('new name', trained_list[i], new_name)
            elif 'res_layer.2' in trained_list[i]:
                new_name = trained_list[i].split('res_layer.2')[0] + 'res_layer.3' + \
                           trained_list[i].split('res_layer.2')[-1]
                # print('new name', trained_list[i], new_name)
            elif 'res_layer.3' in trained_list[i]:
                new_name = trained_list[i].split('res_layer.3')[0] + 'res_layer.4' + \
                           trained_list[i].split('res_layer.3')[-1]
                print('new name', trained_list[i], new_name)
            elif 'res_layer.4' in trained_list[i]:
                new_name = trained_list[i].split('res_layer.4')[0] + 'res_layer.6' + \
                           trained_list[i].split('res_layer.4')[-1]
                print('new name', trained_list[i], new_name)
            else:
                new_name = trained_list[i]

            dict_new[new_name] = dict_trained[trained_list[i]]

        model.load_state_dict(dict_new)
        model = model.to(DEVICE)

        model_ori = IR_50([112, 112]),
        model_ori = model_ori[0]

        model_ori.load_state_dict(torch.load(model_root))
        model_ori = model_ori.to(DEVICE)

    else:
        model = IR_50([112, 112]),
        model = model[0]
        model.load_state_dict(torch.load(model_root))
        model = model.to(DEVICE)

    # make the output dir of privacy masks
    if not os.path.exists(args.adv_out):
        os.makedirs(args.adv_out)

    # load the list of images
    img_list = open(args.target_lst)
    files = img_list.readlines()
    print(len(files))


    i = 0
    np.random.seed(0)
    while i < args.end_id:
      if i < args.start_id:
          i += 1
          continue
      else:
        print(i)
        IMG = np.ones((args.num_shot, 3, 112, 112), dtype='float32') # each identity has 10 training images
        for j in range(args.num_shot):
            name = files[i*args.batch_size+j] # use args.batch_size for generation
            img_name = os.path.join(args.data_dir, name)
            img_name = img_name.split('\n')[0]
            print(i, j, "target", img_name)
            img = cv2.imread(img_name)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = np.transpose(img, (2, 0, 1))
            IMG[j, :, :, :] = img

        IMG = torch.from_numpy(IMG)
        IMG = IMG.to(DEVICE)

        if args.base == 1:
            fim = FIM(args.round, args.alpha, args.step_size, True, args.loss_type, args.nter, args.upper, args.lower)
            noise = fim.process(model, IMG) #use the attack function
        else: #with momentum and DFANet
            mfim = DFANet_MFIM(args.round, args.alpha, args.step_size, True, args.loss_type, args.nter, args.upper, args.lower)
            noise = mfim.process(model, model_ori, IMG) #use the attack function

        noise_j = noise[0].cpu().detach().numpy()
        savenpy = os.path.join(args.adv_out, 'mask_id%d.npy'% i)
        np.save(savenpy, noise_j)

        i += 1

    time2 = datetime.datetime.now()
    print("time consumed: ", time2 - time1)


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained', help='surrogate model', default='../results/IR_50-ArcFace-casia/Backbone_IR_50_Epoch_73_Batch_138000_Time_2020-05-07-23-48_checkpoint.pth')
    parser.add_argument('--adv_out', help='output dir of privacy masks', default='./test')
    parser.add_argument('--target_lst', help='list of training images', default='../list/privacy_train_v3_10.lst')
    parser.add_argument('--data_dir', help='dir of training images', default='/ssd/')
    parser.add_argument('--batch_size', type=int, help='number of samples to generate a mask', default=10)
    parser.add_argument('--num_shot', type=int, help='each identity has 10 training images', default=10)
    parser.add_argument('--nter', type=int, help='initial iterations of convexhull', default=100)
    parser.add_argument('--upper', type=float, help='upper bound of reducedhull', default=1.0)
    parser.add_argument('--lower', type=float, help='lower bound of reducedhull', default=0.0)
    parser.add_argument('--loss_type', type=int, help='type of approximation method:0-->FI-UAP; '
                                                                 '2-->FI-UAP+; 7-->OPOM-ClassCenter; 8-->OPOM-AffineHull;'
                                                                 '9-->OPOM-ConvexHull', default=9)
    parser.add_argument('--alpha', type=float, help='perturbation budeget', default=8)
    parser.add_argument('--step_size', type=float, help='gradient step size, defalt 1 in this work', default=1)
    parser.add_argument('--round', type=int, help='training iterations', default=50)
    parser.add_argument('--base', type=int, help='without transferability enhancement methods', default=0)
    parser.add_argument('--DFANet', type=int, help='whether to use Momentum and DFANet', default=0)
    parser.add_argument('--droprate', type=float, help='dropout rate of DFANet, 0.1 for resnet50 models', default=0.1)
    parser.add_argument('--start_id', type=int, help='start identity', default=-1)
    parser.add_argument('--end_id', type=int, help='end identity', default=500)
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
