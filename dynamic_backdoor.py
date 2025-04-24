import argparse
import os
import pathlib
import re
import time
import datetime
import numpy as np
import torchattacks
from sklearn import metrics
import seaborn as sns
import matplotlib.pyplot as plt
import torch.nn.functional as F
from art.defences.detector.poison import SpectralSignatureDefense, ActivationDefence
from art.estimators.classification.pytorch import PyTorchClassifier
import json

import pandas as pd
import torch
from torch.utils.data import DataLoader

from dataset import build_poisoned_training_set, build_reverse_poisoned_training_set, build_clean_training_set, build_testset
from deeplearning import evaluate_badnets, optimizer_picker, train_one_epoch, reverse_train_model
from models import BadNet

parser = argparse.ArgumentParser(description='Reproduce the basic backdoor attack in "Badnets: Identifying vulnerabilities in the machine learning model supply chain".')
parser.add_argument('--dataset', default='MNIST', help='Which dataset to use (MNIST or CIFAR10, default: MNIST)')
parser.add_argument('--nb_classes', default=10, type=int, help='number of the classification types')
parser.add_argument('--load_local', action='store_true', help='train model or directly load model (default true, if you add this param, then load trained local model to evaluate the performance)')
parser.add_argument('--loss', default='mse', help='Which loss function to use (mse or cross, default: mse)')
parser.add_argument('--optimizer', default='sgd', help='Which optimizer to use (sgd or adam, default: sgd)')
parser.add_argument('--epochs', default=100, help='Number of epochs to train backdoor model, default: 100')
parser.add_argument('--batch_size', type=int, default=64, help='Batch size to split dataset, default: 64')
parser.add_argument('--num_workers', type=int, default=0, help='Batch size to split dataset, default: 64')
parser.add_argument('--lr', type=float, default=0.01, help='Learning rate of the model, default: 0.001')
parser.add_argument('--download', action='store_true', help='Do you want to download data ( default false, if you add this param, then download)')
parser.add_argument('--data_path', default='./data/', help='Place to load dataset (default: ./dataset/)')
parser.add_argument('--device', default='cpu', help='device to use for training / testing (cpu, or cuda:1, default: cpu)')
# poison settings
parser.add_argument('--poisoning_rate', type=float, default=0.1, help='poisoning portion (float, range from 0 to 1, default: 0.1)')
parser.add_argument('--trigger_label', type=int, default=1, help='The NO. of trigger label (int, range from 0 to 10, default: 0)')
parser.add_argument('--trigger_path', default="./triggers/trigger_white.png", help='Trigger Path (default: ./triggers/trigger_white.png)')
parser.add_argument('--trigger_size', type=int, default=5, help='Trigger Size (int, default: 5)')
parser.add_argument('--GPU_mode', action='store_true', help='Using GPUs')

args = parser.parse_args()

def calculate_metrics(TP, FP, TN, FN):
    TPR = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    FPR = FP / (FP + TN) if (FP + TN) > 0 else 0.0
    TNR = TN / (TN + FP) if (TN + FP) > 0 else 0.0
    FNR = FN / (TP + FN) if (TP + FN) > 0 else 0.0

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (TP + TN) / (TP + FP + TN + FN) if (TP + FP + TN + FN) > 0 else 0.0

    return {
        "TPR": TPR,
        "FPR": FPR,
        "TNR": TNR,
        "FNR": FNR,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1_score,
        "Accuracy": accuracy,
    }

def adversarial_learning(model,data,y, attack_type,nb_classes,Targeted=False):

    hyper_param=None
    flag_mis=False
    input_shape = data.shape[1:]
    if attack_type=='FGSM':
        epsilon = np.arange(0.001,2,0.001)


        for eps in epsilon:
            atk= torchattacks.FGSM(model, eps=eps)
            adv_images = atk(data, y)
            pred=model.forward(adv_images)
            if not (y.equal(pred.max(1)[1].cpu())):
                flag_mis=True
                hyper_param=eps
                break

    elif attack_type=='IFGSM':

        atk= torchattacks.FGSM(model, eps=0.0001)
        adv_images=data.clone()
        for i in range(10000):
            adv_images = atk(adv_images, y)
            pred=model.forward(adv_images)
            if not (y== pred.max(1)[1].cpu()):
                flag_mis=True
                break
    elif attack_type== 'DeepFool':
        overshoot=np.arange(0.0001,10,0.0002)
        for ov in overshoot:
            atk=torchattacks.DeepFool(model, steps=250, overshoot=ov)
            adv_images = atk(data, y)
            pred=model.forward(adv_images)
            if not (y== pred.max(1)[1].cpu()):
                flag_mis=True
                hyper_param=ov
                break

    elif attack_type=='CW':
        CWs=np.arange(0.0001,10,0.0002)
        for c_cw in CWs:
            atk= torchattacks.CW(model, c=c_cw,kappa=50, steps=250)
            adv_images = atk(data, y)
            pred=model.forward(adv_images)

            if not (y== pred.max(1)[1].cpu()):
                flag_mis=True
                hyper_param=c_cw
                break
    elif attack_type == 'blackbox':
        """
          max_iter: int = 50,
        max_eval: int = 10000,
        init_eval: int = 100,
        init_size: int = 100,
        Create a HopSkipJump attack instance.
        :param classifier: A trained classifier.
        :param batch_size: The size of the batch used by the estimator during inference.
        :param targeted: Should the attack target one specific class.
        :param norm: Order of the norm. Possible values: "inf", np.inf or 2.
        :param max_iter: Maximum number of iterations.
        :param max_eval: Maximum number of evaluations for estimating gradient.
        :param init_eval: Initial number of evaluations for estimating gradient.
        :param init_size: Maximum number of trials for initial generation of adversarial examples.
        :param verbose: Show progress bars.
        """

        from art.attacks.evasion import HopSkipJump
        from art.estimators.classification import PyTorchClassifier
        import torch.nn as nn
        import torch.nn.functional as F
        import torch.optim as optim
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        classifier = PyTorchClassifier(model=model,loss=criterion, optimizer=optimizer,
            input_shape=input_shape,nb_classes=nb_classes, clip_values=(0, 1))
        adv_crafter = HopSkipJump(classifier,batch_size=len(data),init_eval=10,
            init_size=100,max_iter=20,max_eval=1000,verbose=False)
        adv_images= adv_crafter.generate(data.detach().cpu().numpy())
        if(args.GPU_mode == True):
            adv_images=torch.from_numpy(adv_images).cuda()
        else:
            adv_images=torch.from_numpy(adv_images)

        pred=model(adv_images)
        pred= pred.max(1)[1].cpu()
        flag_mis=False if pred==y else True


    elif attack_type == 'guassian_noise':
        flag_mis=False
        for sigma in np.arange(0.0001,0.0025,0.0001):
            if(args.GPU_mode):
                noise=torch.normal(0, sigma, size=(100, data.shape[1], data.shape[2], data.shape[3])).cuda()
            else:
                noise=torch.normal(0, sigma, size=(100, data.shape[1], data.shape[2], data.shape[3]))
            x = torch.clip(data + noise, 0,1)
            pred=model(x)
            pred= pred.max(1)[1].cpu()
            idx=[i for i,p in enumerate(pred) if p.cpu()==y.cpu()]
            if len(idx)>0:
                delta=[np.sqrt(np.sum(np.square(n.detach().cpu().numpy()))) for n in noise]
                i=np.argmin(np.array(delta))
                adv_images=torch.clip(data+noise[i:i+1],0,1)

                hyper_param=sigma
                flag_mis=True



    if not flag_mis:
        adv_images=data

    return adv_images, flag_mis, hyper_param

def Detect_trojan_input_via_adversarial_learning(backdoor_model, dataset, model_type, adv_method):
    torch.backends.cudnn.deterministic = True

    dataset_train, num_classes = build_poisoned_training_set(is_train=True, dynamic=True, args=args)
    dataset_val_clean, dataset_val_poisoned = build_testset(is_train=False, dynamic=True, mode="Verification", args=args)

    data_loader_val_clean    = DataLoader(dataset_val_clean,     batch_size=1, shuffle=True)
    data_loader_val_poisoned = DataLoader(dataset_val_poisoned,  batch_size=1, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # if you're using MBP M1, you can also use "mps"
    model = BadNet(input_channels = dataset_train.channels, output_num = args.nb_classes).to(device)
    basic_model_path = "./dynamic_checkpoints/badnet-{}-{}.pth".format(dataset, model_type)
    print("## Load model from : %s" % basic_model_path)
    if(args.GPU_mode == True):
        model = model.cuda()
    model.load_state_dict(torch.load(basic_model_path), strict=True)
    model = model.eval()

    N = 200
    dict_adv = {'data':[], 'adv':[],'Trojaned':[], 'features':[],  'true_y':[], 'delta':[], 'hyperP':[], 'success':[] }

    total = 0
    for data,y in data_loader_val_clean:

        if total >= N: 
            break
        if(args.GPU_mode == True):
            data = data.cuda()
        pred = model.forward(data)

        if total % 50 == 0 and total < N:
            print(total)

        if total<N:
            adv_images, flag_mis, hyper_param = adversarial_learning(model, data, y, adv_method, num_classes, Targeted=False)
            if flag_mis == True:
                dict_adv['Trojaned'].append(0)
                total += 1
                dict_adv['hyperP'].append(hyper_param)
                dict_adv['adv'].append(adv_images.detach().cpu())
                a = data.detach().cpu().numpy().squeeze()
                b = adv_images.detach().cpu().numpy()
                dict_adv['delta'].append(np.sqrt(np.sum(np.square(a-b))))

    total = 0
    for data, y in data_loader_val_poisoned:
        if total >= N: 
            break
        
        if(args.GPU_mode == True):
            data = data.cuda()
        pred = model.forward(data)

        if total % 50 == 0 and total < N:
            print(total)

        if total < N:
            adv_images, flag_mis, hyper_param = adversarial_learning(model, data, y, adv_method, num_classes, Targeted=False)
            if flag_mis == True:
                dict_adv['Trojaned'].append(1)
                total += 1
                dict_adv['hyperP'].append(hyper_param)
                dict_adv['adv'].append(adv_images.detach().cpu())
                a = data.detach().cpu().numpy().squeeze()
                b = adv_images.detach().cpu().numpy()
                dict_adv['delta'].append(np.sqrt(np.sum(np.square(a-b))))

    print('*'*30)
    a = [i for i in dict_adv['delta'][ : N] if i is not None]
    b = [i for i in dict_adv['delta'][N : ] if i is not None]
    print('For model: ' , model_type , ' Delta values Average for Clean and Trojan Samples:', np.mean(a), np.mean(b))

    np.save('./dynamic_logs/%s_%s_%s_%s.npy'%(backdoor_model, dataset, model_type, adv_method), dict_adv)

    delta = np.array(dict_adv['delta'])
    Trojaned = np.array(dict_adv['Trojaned'])
    idx_clean = np.where(Trojaned == 0)[0]
    idx_trojan = np.where(Trojaned == 1)[0]

    clean = delta[idx_clean]
    clean.sort()
    trojan = delta[idx_trojan]
    trojan.sort()

    t_tnr = 0.85
    idx = int(t_tnr * len(clean) + 1)
    upper_threshold = clean[idx]

    TP = len([i for i in trojan if i > upper_threshold])
    FP = len([i for i in clean if i > upper_threshold])
    TN = len([i for i in clean if i < upper_threshold])
    FN = len([i for i in trojan if i < upper_threshold])

    metrics = calculate_metrics(TP,FP, TN, FN)

    with open("dynamic_results/" + backdoor_model + "-" + dataset + "-"  + adv_method + ".log", "a") as f:
        f.write("Model {}:\n".format(model_type))
        f.write("    Delta values Average for Clean and Trojan Samples: {}, {}\n".format(np.mean(a), np.mean(b)))
        for keyword in metrics:
            f.write("    {} is {}\n".format(keyword, metrics[keyword]))
        f.write("\n\n\n")

    print('For model: ' , model_type , ' Delta values Average for Clean and Trojan Samples:', np.mean(a), np.mean(b))
    for keyword in metrics:
        print("    {} is {}".format(keyword, metrics[keyword]))

def unpatch_backdoor_model(target_ASR):
    dataset_reverse_train, args.nb_classes = build_reverse_poisoned_training_set(is_train=True, dynamic=True, args=args)
    dataset_val_clean, dataset_val_poisoned = build_testset(is_train=False, dynamic=True, mode="Test", args=args)
    dataset_train, args.nb_classes = build_poisoned_training_set(is_train=True, dynamic=True, args=args)
    clean_dataset_train, _ = build_clean_training_set(is_train=True, args=args)
    data_loader_clean_train  = DataLoader(clean_dataset_train,   batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    data_loader_reverse_train = DataLoader(dataset_reverse_train, batch_size=32, shuffle=False, num_workers=args.num_workers)
    # data_loader_train = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    data_loader_val_clean     = DataLoader(dataset_val_clean,     batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    data_loader_val_poisoned  = DataLoader(dataset_val_poisoned,  batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BadNet(input_channels=dataset_reverse_train.channels, output_num = args.nb_classes).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optimizer_picker(args.optimizer, model.parameters(), lr=args.lr)

    basic_model_path = "./dynamic_checkpoints/badnet-%s-100ASR.pth" % args.dataset
    print("## Load model from : %s" % basic_model_path)
    model.load_state_dict(torch.load(basic_model_path), strict=True)

    ASR = 1
    epoch = 0
    last_ASR = 0
    epoch_interval = 8
    start = 0
    clean_start = 0
    while(ASR > target_ASR):
        last_ASR = ASR
        epoch += 1
        train_stats = reverse_train_model(data_loader_reverse_train, model, criterion, optimizer, args.loss, device, start , start + epoch_interval)
        test_stats = evaluate_badnets(data_loader_val_clean, data_loader_val_poisoned, model, device)
        print(f"# EPOCH {epoch}   loss: {train_stats['loss']:.4f} Test Acc: {test_stats['clean_acc']:.4f}, ASR: {test_stats['asr']:.4f}\n")
        ASR = test_stats['asr']
        ACC = test_stats['clean_acc']
        start = start + epoch_interval
        if(last_ASR - ASR > 0.02):
            # epoch_interval = 1
            epoch_interval = max(1, int(epoch_interval / 2))
        if(last_ASR - ASR < 0.02 and epoch_interval < 8):
            epoch_interval *= 2
        # if(ACC < 0.95):
        #     reverse_train_model(data_loader_clean_train, model, criterion, optimizer, args.loss, device, clean_start , clean_start + 1)
        #     clean_start = clean_start + 1
        # if(ASR < 0.7 and ACC > 0.95):
        if(ASR < 0.7):
            torch.save(model.state_dict(), "./dynamic_checkpoints/badnet-{}-{}ASR.pth".format(args.dataset, int(ASR*100)))

def train_model():
    print("{}".format(args).replace(', ', ',\n'))

    if re.match('cuda:\d', args.device):
        cuda_num = args.device.split(':')[1]
        os.environ['CUDA_VISIBLE_DEVICES'] = cuda_num
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # if you're using MBP M1, you can also use "mps"

    # create related path
    pathlib.Path("./dynamic_checkpoints/").mkdir(parents=True, exist_ok=True)
    pathlib.Path("./dynamic_logs/").mkdir(parents=True, exist_ok=True)

    print("\n# load dataset: %s " % args.dataset)
    dataset_train, args.nb_classes = build_poisoned_training_set(is_train=True, dynamic=True, args=args)
    if args.load_local:
        dataset_val_clean, dataset_val_poisoned = build_testset(is_train=False, dynamic=True, mode="Test", args=args)
    else:
        dataset_val_clean, dataset_val_poisoned = build_testset(is_train=False, dynamic=True, mode="Verification", args=args)
    
    data_loader_train        = DataLoader(dataset_train,         batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    data_loader_val_clean    = DataLoader(dataset_val_clean,     batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    data_loader_val_poisoned = DataLoader(dataset_val_poisoned,  batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    model = BadNet(input_channels=dataset_train.channels, output_num=args.nb_classes).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optimizer_picker(args.optimizer, model.parameters(), lr=args.lr)

    basic_model_path = "./dynamic_checkpoints/badnet-%s-100ASR.pth" % args.dataset
    start_time = time.time()
    if args.load_local:
        print("## Load model from : %s" % basic_model_path)
        model.load_state_dict(torch.load(basic_model_path), strict=True)
        test_stats = evaluate_badnets(data_loader_val_clean, data_loader_val_poisoned, model, device)
        print(f"Test Clean Accuracy(TCA): {test_stats['clean_acc']:.4f}")
        print(f"Attack Success Rate(ASR): {test_stats['asr']:.4f}")
    else:
        print(f"Start training for {args.epochs} epochs")
        stats = []
        next_data_loader = data_loader_train
        for epoch in range(args.epochs):
            train_stats = train_one_epoch(next_data_loader, model, criterion, optimizer, args.loss, device)
            test_stats = evaluate_badnets(data_loader_val_clean, data_loader_val_poisoned, model, device)
            print(f"# EPOCH {epoch}   loss: {train_stats['loss']:.4f} Test Acc: {test_stats['clean_acc']:.4f}, ASR: {test_stats['asr']:.4f}\n")
            
            # save model 
            torch.save(model.state_dict(), basic_model_path)

            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                            **{f'test_{k}': v for k, v in test_stats.items()},
                            'epoch': epoch,
            }

            # save training stats
            stats.append(log_stats)
            df = pd.DataFrame(stats)
            df.to_csv("./dynamic_logs/%s_trigger%d.csv" % (args.dataset, args.trigger_label), index=False, encoding='utf-8')

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

def model_testing(model, x, y, test_samples):
    succ=0

    if(args.GPU_mode == True):
        test_samples = torch.stack(test_samples).cuda()
    else:
        test_samples = torch.stack(test_samples)

    temp = (x + test_samples)/2
    pred = model(temp)
    pred = pred.max(1)[1].cpu()
    succ = np.sum([1 for i in range(len(pred)) if y == pred[i]])

    return succ / len(test_samples)

def Detect_trojan_input_via_STRIP(backdoor_model, dataset, model_type):
    N = 1000

    dataset_train, args.nb_classes = build_poisoned_training_set(is_train=True, dynamic=True, args=args)
    clean_dataset_train, _ = build_clean_training_set(is_train=True, args=args)
    dataset_val_clean, dataset_val_poisoned = build_testset(is_train=False, dynamic=True, mode="Verification", args=args)
    
    data_loader_val_clean    = DataLoader(dataset_val_clean,     batch_size = 1, shuffle=False, num_workers=args.num_workers)
    data_loader_val_poisoned = DataLoader(dataset_val_poisoned,  batch_size = 1, shuffle=False, num_workers=args.num_workers)

    idx=np.random.choice(len(clean_dataset_train), 20)

    test_samples=[clean_dataset_train[i][0] for i in idx]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # if you're using MBP M1, you can also use "mps"
    model = BadNet(input_channels = dataset_train.channels, output_num = args.nb_classes).to(device)
    basic_model_path = "./dynamic_checkpoints/badnet-{}-{}.pth".format(dataset, model_type)
    print("## Load model from : %s" % basic_model_path)
    if(args.GPU_mode == True):
        model = model.cuda()
    model.load_state_dict(torch.load(basic_model_path), strict=True)
    model = model.eval()

    detection=np.zeros((2,N))
    for T,loader in enumerate([data_loader_val_clean, data_loader_val_poisoned]):
        total=0
        for data,y in loader:
            if total%50==0 and total < N:
                print(total)
            if total >= N: 
                break
            if(args.GPU_mode == True):
                data = data.cuda()
            if total < N:
                detection[T,total] = model_testing(model, data, y, test_samples)
                total+=1

    TP = np.sum([1 for i in detection[1,:] if i >= 0.5]) 
    FP = np.sum([1 for i in detection[0,:] if i >= 0.5]) 
    TN = np.sum([1 for i in detection[0,:] if i < 0.5])
    FN = np.sum([1 for i in detection[1,:] if i < 0.5]) 

    metrics = calculate_metrics(TP, FP, TN, FN)

    with open("dynamic_results/" + backdoor_model + "-" + dataset + "-STRIP.log", "a") as f:
        f.write("Model {}:\n".format(model_type))
        for keyword in metrics:
            f.write("    {} is {}\n".format(keyword, metrics[keyword]))
        f.write("\n\n\n")

    for keyword in metrics:
        print("    {} is {}".format(keyword, metrics[keyword]))

def Detect_trojan_input_via_SpecSig_AC(backdoor_model, dataset, model_type, defense):

    dataset_train, args.nb_classes = build_poisoned_training_set(is_train=True, dynamic=True, args=args)
    dataset_val_clean, dataset_val_poisoned = build_testset(is_train=False, dynamic=True, mode="Verification", args=args)

    dataset_val_clean_x = torch.cat([torch.unsqueeze(xy[0],dim=0) for xy in dataset_val_clean],dim=0).numpy()
    dataset_val_poisoned_x = torch.cat([torch.unsqueeze(xy[0],dim=0) for xy in dataset_val_poisoned],dim=0).numpy()

    dataset_all_x = np.concatenate([dataset_val_clean_x, dataset_val_poisoned_x], axis = 0)

    y_clean=[xy[1] for xy in dataset_val_clean]
    y_test_all=np.array(y_clean+[args.trigger_label]*len(dataset_val_poisoned_x))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # if you're using MBP M1, you can also use "mps"
    model = BadNet(input_channels = dataset_train.channels, output_num = args.nb_classes).to(device)
    basic_model_path = "./dynamic_checkpoints/badnet-{}-{}.pth".format(dataset, model_type)
    print("## Load model from : %s" % basic_model_path)
    if(args.GPU_mode == True):
        model = model.cuda()
    model.load_state_dict(torch.load(basic_model_path), strict=True)
    model = model.eval()

    loss= F.cross_entropy
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=0.0005)

    p_classifier =  PyTorchClassifier(model, loss, (3,32,32), args.nb_classes, optimizer)

    N = len(dataset_val_clean_x)

    is_clean = np.array([1]*N+[0]*N)
    if defense == "SpecSig":
        Spec = SpectralSignatureDefense(p_classifier, dataset_all_x, y_test_all,
                                      expected_pp_poison=0.5,batch_size=args.batch_size)

        conf_matrix= Spec.evaluate_defence(is_clean)
        conf_matrix = json.loads(conf_matrix)

        TP, TN, FP, FN=0, 0, 0, 0
        for key in conf_matrix:
            dict_info=conf_matrix[key]
            TP += int(dict_info["TruePositive"]["numerator"])
            TN += int(dict_info["TrueNegative"]["numerator"])
            FP += int(dict_info["FalsePositive"]["numerator"])
            FN += int(dict_info["FalseNegative"]["numerator"])

        metrics = calculate_metrics(TP, FP, TN, FN)

        with open("dynamic_results/" + backdoor_model + "-" + dataset + "-" + defense + ".log", "a") as f:
            f.write("Model {}:\n".format(model_type))
            for keyword in metrics:
                f.write("    {} is {}\n".format(keyword, metrics[keyword]))
            f.write("\n\n\n")

        for keyword in metrics:
            print("    {} is {}".format(keyword, metrics[keyword]))
    
    valid_reduce = ["FastICA", "PCA"]
    valid_analysis = ["smaller", "distance"]
    nb_dims = [2,10]

    if defense == "AC":
        for reduce in valid_reduce:
            for analysis  in valid_analysis:
                for nb_d in nb_dims:
                    kwargs = {"reduce":reduce , "analysis": analysis, "nb_dim": nb_d}
                    AC=ActivationDefence(p_classifier, dataset_all_x, y_test_all)
                    AC.detect_poison(nb_dims=kwargs["nb_dim"],reduce=  kwargs["reduce"],
                                cluster_analysis= kwargs["analysis"], nb_clusters=2)
                    conf_matrix= AC.evaluate_defence(is_clean)

                    conf_matrix = json.loads(conf_matrix)

                    TP, TN, FP, FN=0, 0, 0, 0
                    for key in conf_matrix:
                        dict_info=conf_matrix[key]
                        TP += int(dict_info["TruePositive"]["numerator"])
                        TN += int(dict_info["TrueNegative"]["numerator"])
                        FP += int(dict_info["FalsePositive"]["numerator"])
                        FN += int(dict_info["FalseNegative"]["numerator"])

                    metrics = calculate_metrics(TP, FP, TN, FN)

                    with open("dynamic_results/" + backdoor_model + "-" + dataset + "-" + defense + ".log", "a") as f:
                        f.write("Model {}:\n".format(model_type))
                        f.write("   Reduce: {}:\n".format(reduce))
                        f.write("   Analysis {}:\n".format(analysis))
                        f.write("   nb_dims {}:\n".format(nb_d))
                        for keyword in metrics:
                            f.write("    {} is {}\n".format(keyword, metrics[keyword]))
                        f.write("\n\n\n")

                    for keyword in metrics:
                        print("    {} is {}".format(keyword, metrics[keyword]))

def BatchRobustnessAnalysis():
    adversarial_methods = ['FGSM', 'IFGSM']
    model_types = ["clean-model", "23ASR", "32ASR", "42ASR", "53ASR", "66ASR", "100ASR"]
    for model_type in model_types:
        # Detect_trojan_input_via_SpecSig_AC("Badnets", "MNIST", model_type, "AC")
        # Detect_trojan_input_via_SpecSig_AC("Badnets", "MNIST", model_type, "SpecSig")
        # Detect_trojan_input_via_STRIP("Badnets", "MNIST", model_type)
        for adversarial_method in ['FGSM', 'IFGSM']:
            Detect_trojan_input_via_adversarial_learning("Badnets", "MNIST", model_type, adversarial_method)

if __name__ == "__main__":
    Detect_trojan_input_via_adversarial_learning("Badnets", "MNIST", "66ASR", "FGSM")
    # train_model()
    # unpatch_backdoor_model(0.3)
    # Detect_trojan_input_via_SpecSig_AC("Badnets", "MNIST", "100ASR", "AC")
    # BatchRobustnessAnalysis()
    # unpatch_backdoor_model(0.3)
    # main()
    # RobustnessAnalysis("MNIST", "clean-model")