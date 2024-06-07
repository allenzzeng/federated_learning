import argparse, json
import datetime
import os
import logging
import torch
import random
from predictcopy import falsePredict #自己写的用于找出困难样本的代码


from server import *
from client import *
import models, datasets

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Federated Learning')
    try:
        parser.add_argument('-c', '--conf', dest='conf')
        args = parser.parse_args()

        with open(args.conf, 'r') as f:
            conf = json.load(f)
    except:
        conf_file = './utils/conf.json'
        with open(conf_file, 'r') as f:
            conf = json.load(f)

    train_datasets, eval_dataset = datasets.get_dataset()

    server = Server(conf, eval_dataset)
    clients = []

    # for c in range(conf["no_models"]):
    #     clients.append(Client(conf, server.global_model, train_datasets, c))
    for c in range(len(train_datasets)):
        clients.append(Client(conf, server.global_model, train_datasets[c], c))
        
    print("\n\n")

    valtimelist=[]
    Valid_Accuracy_list=[]
    best_acc=0
    alist=[]

    for e in range(conf["global_epochs"]):

        # candidates = random.sample(clients, conf["k"])
        
        weight_accumulator = {}

        for name, params in server.global_model.state_dict().items():
            weight_accumulator[name] = torch.zeros_like(params)

        for c in clients:
            diff = c.local_train(server.global_model)

            for name, params in server.global_model.state_dict().items():
                weight_accumulator[name].add_(diff[name])

        server.model_aggregate(weight_accumulator)

        server.model_eval(e,best_acc,valtimelist,Valid_Accuracy_list)
        
        # acc, loss = server.model_eval()
        # print("Epoch %d, acc: %f, loss: %f\n" % (e, acc, loss))


        save_path = os.path.join(os.getcwd(), 'weight', 'global_model_epoch_{}.pt'.format(e))
        torch.save(server.global_model.state_dict(), save_path)

        # if e>=0:
        #     # torch.save(server.global_model.state_dict(), 'global_model_epoch_{}.pt'.format(e))
        #     #成员变量alist在上面
        #     alist=falsePredict(alist,e)
        #     # print(alist)

    import matplotlib.pyplot as plt
    x1 = range(0, conf["global_epochs"])
    y1 = Valid_Accuracy_list
    plt.subplot(1, 1, 1)
    plt.plot(x1,y1,'bx-',label = 'Valid Accuracy')
    plt.text(best_acc,Valid_Accuracy_list[best_acc],'%.3f'%Valid_Accuracy_list[best_acc],ha='center',va='bottom')
    plt.xlabel('vs. epoch')
    plt.ylabel('Valid Accuracy')
    plt.legend(loc='best')
    plt.show()
    print("done")

        #_____________________________________
    import xlwt
    workbook = xlwt.Workbook()
    sheet = workbook.add_sheet('sheet1')
    # 将数据写入文件,i是enumerate()函数返回的序号数
    for i,e in enumerate(Valid_Accuracy_list):
        sheet.write(i,0,e)
    for i,e in enumerate(valtimelist):
        sheet.write(i,2,e)
    # for i in range(3):
    #     for j in range(3):
    #         sheet.write(i, j+5, str(valcm[i][j]))
    # # 保存文件
    workbook.save('./result_files/acc_time.xls')