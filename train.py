import torch
from torch_geometric.datasets import TUDataset, Entities
from torch_geometric.data import DataLoader
from torch_geometric.data import Batch
import time
import random
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "3"

def train(model, train_dataset, test_dataset = None, max_epoch = 400, lr = 0.01, train_batch_size=128, eval_batch_size = 128, save_model = True, temp_name = 'model_temp'):
    print('prepare dataloader')
    #print('current device: ', torch.cuda.current_device())
    loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=8)
    print('done')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0)
    max_epoch = max_epoch
    best_test_acc = 0.0
    for epoch in range(max_epoch):
        avg_loss = 0.0
        correct = 0.0
        begin_time = time.time()
        total_nodes = 0

        for batch_idx, data in enumerate(loader):
            #print(data.batch)
            optimizer.zero_grad()
            data = data.to(device)
            preds = model.forward(data)
            #print(preds)
            label = data.y.to(device)
            loss = model.loss(preds, label)
            _, indices = torch.max(preds, 1)

            #print(indices.shape)
            #print('===subgraph size: ' + str(data.x.shape))
            #print(preds)
            #print(indices)
            #print(label)

            total_nodes += data.y.shape[0]# if not dataset.subgraph else batch_size
            correct += torch.sum(indices == label)
            #print(data.y.shape[0])
            #print(torch.sum(indices == label))
            loss.backward()
            #for p in model.parameters():
            #    print(p.grad.norm())
            #print('==========================================')
            optimizer.step()

            avg_loss += loss


            #break
        avg_loss /= batch_idx + 1
        acc = correct / total_nodes
        #acc = correct / dataset.len()
        elapsed = time.time() - begin_time
        print("Epoch: ", epoch,"Avg loss: ", avg_loss.cpu().data.numpy(), "; acc: ", acc.cpu().data.numpy(), "; epoch time: ", elapsed)
        if avg_loss < 0.01:
            print('early stop')
            break
        if test_dataset is not None:
            print('eval test...')
            acc = eval(model, test_dataset, eval_batch_size)
            if acc > best_test_acc:
                best_test_acc = acc
                if save_model:
                    print('saving...')
                    torch.save(model, './checkpoint/'+temp_name)
            print('\n')

def eval(model, dataset, batch_size=128):

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    #optimizer = torch.optim.Adam(model.parameters(), lr=0.00005, weight_decay=5e-4)
    max_epoch = 1
    with torch.no_grad():
        for epoch in range(max_epoch):
            avg_loss = 0.0
            correct = 0.0
            begin_time = time.time()
            total_nodes = 0
            for batch_idx, data in enumerate(loader):
                #print(data.batch)
                #optimizer.zero_grad()
                data = data.to(device)
                preds = model.forward(data)
                #print(preds)
                label = data.y.to(device)
                loss = model.loss(preds, label)
                _, indices = torch.max(preds, 1)

                #print(indices.shape)
                #print('===subgraph size: ' + str(data.x.shape))
                #print(preds)
                #print(indices)
                #print(label)

                total_nodes += data.y.shape[0]# if not dataset.subgraph else batch_size
                correct += torch.sum(indices == label)

                #loss.backward()
                #optimizer.step()
                avg_loss += loss


                #break
            avg_loss /= batch_idx + 1
            acc = correct / total_nodes
            #acc = correct / dataset.len()
            elapsed = time.time() - begin_time
            print("Epoch: ", epoch,"Avg loss: ", avg_loss.cpu().data.numpy(), "; acc: ", acc.cpu().data.numpy(), "; epoch time: ", elapsed)
    
    return acc.cpu().data.numpy()




def node_pred_task_train(model, train_dataset, max_epoch = 400, lr = 0.01, train_batch_size=128, save_model = True, temp_name = 'model_temp', train_rate = 0.8):
    print('prepare dataloader')
    loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=8)
    print('done')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0)
    max_epoch = max_epoch
    best_test_acc = 0.0
    best_test_loss = float('inf')

    node_number = train_dataset[0].x.shape[0]
    node_list = [i for i in range(node_number)]
    random.shuffle(node_list)
    train_node_list = node_list[0:int(train_rate*node_number)]
    test_node_list = node_list[int(train_rate*node_number):]

    for epoch in range(max_epoch):
        avg_loss = 0.0
        test_loss = 0.0
        correct = 0.0
        begin_time = time.time()
        total_nodes = 0

        for batch_idx, data in enumerate(loader):
            #print(data.batch)
            optimizer.zero_grad()
            data = data.to(device)
            preds = model.forward(data)
            #print(preds)
            label = data.y.to(device)

            train_preds = preds[train_node_list]
            test_preds = preds[test_node_list]

            train_label = label[train_node_list]
            test_label = label[test_node_list]

            loss = model.loss(train_preds, train_label)
            test_loss += model.loss(test_preds, test_label)
            
            _, train_indices = torch.max(train_preds, 1)
            _, test_indices = torch.max(test_preds, 1)

            #print(indices.shape)
            #print('===subgraph size: ' + str(data.x.shape))
            #print(preds)
            #print(indices)
            #print(label)

            total_nodes += len(train_label)# if not dataset.subgraph else batch_size
            correct += torch.sum(train_indices == train_label)
            #print(data.y.shape[0])
            #print(torch.sum(indices == label))
            loss.backward()
            #for p in model.parameters():
            #    print(p.grad.norm())
            #print('==========================================')
            optimizer.step()

            avg_loss += loss
            test_acc = torch.sum(test_indices == test_label) / len(test_label)


            #break
        avg_loss /= batch_idx + 1
        test_loss /= batch_idx + 1

        acc = correct / total_nodes
        #acc = correct / dataset.len()
        elapsed = time.time() - begin_time
        print("Epoch: ", epoch,"Avg loss: ", avg_loss.cpu().data.numpy(), "; acc: ", acc.cpu().data.numpy(), "; epoch time: ", elapsed)
        if avg_loss < 0.01:
            print('early stop')
            break

        print('eval test...')
        acc = test_acc
        print('test acc: ', acc.cpu().data.numpy())
        print('test loss: ', test_loss.cpu().data.numpy())


        #if acc > best_test_acc:
        #    best_test_acc = acc
        #    if save_model:
        #        print('saving...')
        #        torch.save(model, './checkpoint/'+temp_name)
        
        if test_loss <= best_test_loss:
            best_test_loss = test_loss
            if save_model:
                print('saving...')
                torch.save(model, './checkpoint/'+temp_name)
        
        print('\n')