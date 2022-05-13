import os
import numpy as np
import sklearn
import sklearn.preprocessing
import cv2
import mxnet as mx
import sys
import argparse
import datetime


def get_image_feature(img_dir, img_list_path, pretrained, gpu_id, batch_size, img_type = None, lenth = None,img_per_tmp = None,
                      mask = None):
    img_list = open(img_list_path)
    ctx = mx.gpu(gpu_id)
    vec = pretrained.split(',')
    sym, arg_params, aux_params = mx.model.load_checkpoint(vec[0], int(vec[1]))
    all_layers = sym.get_internals()
    sym = all_layers['fc1_output']
    image_size = (112, 112)
    model = mx.mod.Module(symbol=sym, context=ctx, label_names=None)
    model.bind(for_training=False, data_shapes=[('data', (batch_size, 3, image_size[0], image_size[1]))])
    model.set_params(arg_params, aux_params)

    files = img_list.readlines()
    #print('files:', len(files))
    img_feats = []

    if lenth is not None:
        file_lenth = lenth
    else:
        file_lenth = len(files)
    start = 0
    while True:
        if start % 10000 ==0:
            print("processing", start)
        end = min(start + batch_size, file_lenth)
        if start >= end:
            break
        input_blob = np.zeros((batch_size, 3, image_size[0], image_size[1]), dtype=np.uint8)
        for i in range(start, end):
            img_name = files[i]
            #print(img_name)
            if img_type == 'lfw':
                img_name = img_name.split('/')
                a = img_name[0]
                b = img_name[1].split('\n')[0]
                out_dir = os.path.join(img_dir, "%s" % (a))
                img_name = os.path.join(out_dir, "%s" % (b))
            elif img_type == 'MF2':
                image_name = img_name.split()[1]
                img_name = os.path.join(img_dir, image_name)
            else:
                img_name = os.path.join(img_dir, img_name.split('\n')[0])

            #print(img_name)
            img = cv2.imread(img_name) #bgr
            if img is None:
                print(img_name)

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # rgb

            if mask is not None:
                mask_img_npy = os.path.join(mask, 'mask_id%d.npy'% int(i/img_per_tmp))
                adv_noise = np.load(mask_img_npy) #rgb
                if len(adv_noise.shape) == 4:
                    adv_noise = adv_noise[0]
                adv_noise = np.transpose(adv_noise, (1, 2, 0))
                img_float = img.astype(np.float64)
                adv_img = np.maximum(np.minimum(img_float + adv_noise, 255), 0)
                img = adv_img.astype(dtype=np.uint8)

            img = np.transpose(img, (2, 0, 1))  # 3*112*112, RGB
            input_blob[i-start] = img
        data = mx.nd.array(input_blob)
        db = mx.io.DataBatch(data=(data,))
        model.forward(db, is_train=False)
        feat = model.get_outputs()[0].asnumpy()
        for i in range(end - start):
            fea = feat[i]
            fea = fea.flatten()
            img_feats.append(fea)
        start = end
    img_feats = np.array(img_feats).astype(np.float32)
    img_feats = sklearn.preprocessing.normalize(img_feats)
    return img_feats

def evaluation(gpu_id, query_img_feats_mask, query_img_feats, gallery_noise_feats, img_per_tmp = 10):
    print(query_img_feats_mask.shape)
    print(query_img_feats.shape)
    print(gallery_noise_feats.shape)
    query_img_num = int(query_img_feats_mask.shape[0] / img_per_tmp)
    query_num = query_img_num * img_per_tmp * (img_per_tmp - 1)
    gallery_num = gallery_noise_feats.shape[0]
    print(query_num, gallery_num)

    query_img_feats_mask = mx.nd.array(query_img_feats_mask, ctx = mx.gpu(gpu_id))
    query_img_feats = mx.nd.array(query_img_feats, ctx=mx.gpu(gpu_id))
    gallery_noise_feats = mx.nd.array(gallery_noise_feats, ctx=mx.gpu(gpu_id))

    correct_num_top1 = 0
    correct_num_top5 = 0
    correct_num_top10 = 0
    for id in range(query_img_num):
      for i in range(img_per_tmp):
        for j in range(img_per_tmp):
            if i == j:
                continue
            else:
                query_feat = query_img_feats_mask[id * img_per_tmp + i]
                target_feat = mx.nd.zeros((1,512), ctx=mx.gpu(gpu_id))
                target_feat[0] = query_img_feats[id * img_per_tmp + j]
                #embed()
                gallery_feat = mx.nd.concat(target_feat, gallery_noise_feats, dim=0)

                similarity = mx.nd.dot(query_feat, mx.nd.transpose(gallery_feat))
                top_inds = mx.nd.argsort(-similarity)
                top_inds = top_inds.asnumpy()

                if top_inds[0] == 0:
                    correct_num_top1 += 1

                if 0 in top_inds[0:5]:
                    correct_num_top5 += 1
                if 0 in top_inds[0:10]:
                    correct_num_top10 += 1
    print("acc top1 = %1.5f, protect top1 = %1.5f" % (correct_num_top1 / float(query_num), 1.0-correct_num_top1 / float(query_num)) )
    print("acc top5 = %1.5f, protect top5 = %1.5f" % (correct_num_top5 / float(query_num), 1.0- correct_num_top5 / float(query_num)) )
    print("acc top10 = %1.5f, protect top10 = %1.5f" % (correct_num_top10 / float(query_num), 1.0-correct_num_top10 / float(query_num)) )


def test(query_image_dir, query_test_image_list, test_img_per_id, gallery_noise_dir,
         gallery_noise_list, pretrained, gpu, batch_size, mask = None, distract_lenth = None, gallery_image_type = None):
    query_test_img_feats_mask = get_image_feature(query_image_dir, query_test_image_list, pretrained,
                                              gpu, batch_size, img_type = None, img_per_tmp = test_img_per_id, mask = mask)
    query_test_img_feats = get_image_feature(query_image_dir, query_test_image_list, pretrained,
                                                  gpu, batch_size)

    gallery_noise_feats = get_image_feature(gallery_noise_dir, gallery_noise_list, pretrained,
                                            gpu, batch_size, img_type=gallery_image_type, lenth = distract_lenth)

    evaluation(gpu,query_test_img_feats_mask, query_test_img_feats, gallery_noise_feats, img_per_tmp = test_img_per_id)

def main(args):
    print(args)

    time1 = datetime.datetime.now()
    test(args.query_image_dir, args.query_test_image_list, args.test_img_per_id,
    args.gallery_noise_dir, args.gallery_noise_list,
    args.pretrained, args.gpu, args.batch_size, args.msk_dir, args.lenth, args.gallery_image_type)
    time2 = datetime.datetime.now()
    print("time consumed: ", time2 - time1)

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, help='GPU', default=0)
    parser.add_argument('--batch-size', type=int, help='batch size for feature extraction', default=100)
    parser.add_argument('--lenth', type=int, help='disctractor size', default=10000)
    parser.add_argument('--test_img_per_id', type=int, help='#samples of each identity', default=5)
    parser.add_argument('--gallery_image_type', help='query image type', default='MF2')
    parser.add_argument('--query_image_dir', help='dir of query images', default='/ssd/')
    parser.add_argument('--query_test_image_list', help='list of query images',
                        default='/home/zhongyaoyao/yy/privacy_mxnet/list/privacy_test_v3_5.lst')
    parser.add_argument('--gallery_noise_dir', help='dir of gallery images', default='/ssd/MegaFace/challenge2/Distractor')
    parser.add_argument('--gallery_noise_list', help='list of gallery images', default='/ssd/MegaFace/challenge2/Distractor/lst')
    parser.add_argument('--pretrained', type=str, help='model file', default='../models/r50_webface_arc/model,100')
    parser.add_argument('--msk_dir', help='msk path', default = None)
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))

