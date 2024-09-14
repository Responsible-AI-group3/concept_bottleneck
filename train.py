import pdb
import os
import sys
import argparse

import math
import torch
import numpy as np
from analysis import Logger, AverageMeter, accuracy, binary_accuracy

from dataset import load_data, find_class_imbalance
from models import   ModelXtoY, ModelXtoC, ModelXtoCtoY, ModelCtoY



def run_epoch_CtoY(model, optimizer, loader, criterion, args,device, is_training):
    """
    C -> Y: Predicting class labels using only attributes with MLP for Independent and Sequential models 
    """
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    if is_training:
        model.train()
    else:
        model.eval()
    for _, data in enumerate(loader):
        inputs, labels = data
        if isinstance(inputs, list):
            #inputs = [i.long() for i in inputs]
            inputs = torch.stack(inputs).t().float()
        inputs = torch.flatten(inputs, start_dim=1).float()
        inputs_var = torch.autograd.Variable(inputs).to(device)
        labels_var = torch.autograd.Variable(labels).to(device)

        
        outputs = model(inputs_var)
        loss = criterion(outputs, labels_var)
        acc = accuracy(outputs, labels, topk=(1,))
        loss_meter.update(loss.item(), inputs.size(0))
        acc_meter.update(acc[0].item(), inputs.size(0))

        if is_training:
            optimizer.zero_grad() #zero the parameter gradients
            loss.backward()
            optimizer.step() #optimizer step to update parameters

    return loss_meter, acc_meter


def run_epoch_XtoC(model, optimizer, loader, c_criterion, args,device, is_training):
    """
    For the rest of the networks (X -> A, cotraining, simple finetune)
    """
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    if is_training:
        model.train()
    else:
        model.eval()

    for _, data in enumerate(loader):
        if c_criterion is None:
            inputs, labels = data
            attr_labels, attr_labels_var = None, None
        else:
            inputs, labels, attr_labels = data
            if args.n_attributes > 1:
                attr_labels = [i.long() for i in attr_labels]
                attr_labels = torch.stack(attr_labels).t()#.float() #N x 312
            else:
                if isinstance(attr_labels, list):
                    attr_labels = attr_labels[0]
                attr_labels = attr_labels.unsqueeze(1)
            attr_labels_var = torch.autograd.Variable(attr_labels).float()
            attr_labels_var = attr_labels_var.cuda() if torch.cuda.is_available() else attr_labels_var

        inputs_var = torch.autograd.Variable(inputs)
        inputs_var = inputs_var.to(device)
        labels_var = torch.autograd.Variable(labels)
        labels_var = labels_var.to(device)

        if is_training and args.use_aux:
            outputs, aux_outputs = model(inputs_var)
            losses = []

            for i in range(len(c_criterion)): # Loop over each concept and give an individual loss for each. 
                losses.append(1.0 * c_criterion[i](outputs[:,i], attr_labels_var[:,i]) + 0.4 * c_criterion[i](aux_outputs[:,i], attr_labels_var[:,i]))
        else: #testing or no aux logits
            outputs = model(inputs_var)
            losses = []

            for i in range(len(c_criterion)):
                losses.append(1.0 * c_criterion[i](outputs[:,i], attr_labels_var[:,i]))

        #sigmoid_outputs = torch.nn.Sigmoid()(torch.cat(outputs, dim=1)) #Sigmoid already applied. 
        acc = binary_accuracy(outputs, attr_labels)
        acc_meter.update(acc.data.cpu().numpy(), inputs.size(0))




        total_loss = sum(losses)/ args.n_attributes

        loss_meter.update(total_loss.item(), inputs.size(0))

        if is_training:
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
    return loss_meter, acc_meter

def run_epoch_XtoY(model, optimizer, loader, criterion, args,device, is_training):
    """
    X -> Y: Predicting class labels using only images with CNN for Independent and Sequential models
    """
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    if is_training:
        model.train()
    else:
        model.eval()

    for _, data in enumerate(loader):
        inputs, labels = data #inputs are images, labels are class labels
        
        inputs_var = torch.autograd.Variable(inputs)
        inputs_var = inputs_var.to(device)
        labels_var = torch.autograd.Variable(labels)
        labels_var = labels_var.to(device)

        if is_training and args.use_aux:
            Yhat, aux_Yhat = model(inputs_var)

            #Calculate y loss
            loss = 1.0 * criterion(Yhat, labels_var) + 0.4 * criterion(aux_Yhat, labels_var)

        else: #testing or no aux logits
            Yhat = model(inputs_var)

            #Calculate y loss
            loss = criterion(Yhat, labels_var)

        loss_meter.update(loss.item(), inputs.size(0))

        #Calculate class prediction accuracy
        acc = accuracy(Yhat, labels, topk=(1,)) #only care about class prediction accuracy
        acc_meter.update(acc[0].item(), inputs.size(0))

        if is_training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return loss_meter, acc_meter

def run_epoch_XtoCtoY(model, optimizer, loader, y_criterion, c_criterion, args,device, is_training):
    """
    For the rest of the networks (X -> A, cotraining, simple finetune)
    """
    c_loss_meter = AverageMeter()
    c_acc_meter = AverageMeter()

    y_loss_meter = AverageMeter()
    y_acc_meter = AverageMeter()

    loss_meter = AverageMeter() #total loss


    if is_training:
        model.train()
    else:
        model.eval()

    for _, data in enumerate(loader):


        inputs, labels, attr_labels = data

        # Process the attribute labels
        if args.n_attributes > 1:
            attr_labels = [i.long() for i in attr_labels]
            attr_labels = torch.stack(attr_labels).t()#.float() #N x 312
        else:
            if isinstance(attr_labels, list):
                attr_labels = attr_labels[0]
            attr_labels = attr_labels.unsqueeze(1)
        attr_labels_var = torch.autograd.Variable(attr_labels).float()
        attr_labels_var = attr_labels_var.cuda() if torch.cuda.is_available() else attr_labels_var

        # Process the inputs and labels
        inputs_var = torch.autograd.Variable(inputs)
        inputs_var = inputs_var.cuda() if torch.cuda.is_available() else inputs_var
        labels_var = torch.autograd.Variable(labels)
        labels_var = labels_var.cuda() if torch.cuda.is_available() else labels_var

        if is_training and args.use_aux:
            Chat,Yhat, aux_Chat,aux_Yhat = model(inputs_var)
            losses = []

            #Calculate y loss
            loss_main = 1.0 * y_criterion(Yhat, labels_var) + 0.4 * y_criterion(aux_Yhat, labels_var)
            losses.append(loss_main)

            #Calculate the atribute loss by looping over each prediction. 
            for i in range(len(c_criterion)):
                losses.append(args.attr_loss_weight * (1.0 * c_criterion[i](Chat[:, i], attr_labels_var[:, i]) + 0.4 * c_criterion[i](aux_Chat[:, i], attr_labels_var[:, i])))

        else: #testing or no aux logits
            Chat,Yhat = model(inputs_var)
            losses = []

            #Calculate y loss
            loss_main = y_criterion(Yhat, labels_var)
            losses.append(loss_main)

            for i in range(len(c_criterion)):
                    losses.append(args.attr_loss_weight * c_criterion[i](Chat[:, i], attr_labels_var[:, i]))


        #Calculate concept prediction accuracy
        #sigmoid_outputs = torch.nn.Sigmoid()(torch.cat(outputs, dim=1))
        acc = binary_accuracy(Chat, attr_labels)
        c_acc_meter.update(acc.data.cpu().numpy(), inputs.size(0))

        #Calculate class prediction accuracy
        acc = accuracy(Yhat, labels, topk=(1,)) #only care about class prediction accuracy
        y_acc_meter.update(acc[0], inputs.size(0))

        #cotraining, loss by class prediction and loss by attribute prediction have the same weight
        total_loss = losses[0] + sum(losses[1:])
        if args.normalize_loss:
            total_loss = total_loss / (1 + args.attr_loss_weight * args.n_attributes)
        
        y_loss_meter.update(losses[0].item(), inputs.size(0))
        c_loss_meter.update(sum(losses[1:]).item(), inputs.size(0))
        loss_meter.update(total_loss.item(), inputs.size(0))
        if is_training:
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

        
    return loss_meter,c_loss_meter,y_loss_meter, c_acc_meter,y_acc_meter 

def train(model,train_mode, args):
    
    # Determine imbalance for concepts
    if train_mode == "X_to_C" or train_mode == "X_to_C_to_Y":
        train_data_path = os.path.join(args.data_dir, 'train.pkl')
        if args.weighted_loss == 'multiple':
            imbalance = find_class_imbalance(train_data_path, True)
        else:
            imbalance = find_class_imbalance(train_data_path, False)
    else:
        imbalance = None

    device = torch.device(args.device)

    #Define logger
    logger = Logger(os.path.join(args.log_dir, 'log.txt')) # log file for the main task
    train_logger = Logger(os.path.join(args.log_dir, 'train_log.txt')) # log file for plotting
    logger.write(str(args) + '\n')
    logger.write(str(imbalance) + '\n')
    logger.flush()

    model = model.to(device)

    #Define the Y loss function
    if train_mode == "C_to_Y" or train_mode == "X_to_C_to_Y" or train_mode == "X_to_Y":
        y_criterion = torch.nn.CrossEntropyLoss()

    #Define the concept loss function
    if train_mode == "X_to_C" or train_mode == "X_to_C_to_Y":
        c_criterion = [] #separate criterion (loss function) for each attribute

        if args.weighted_loss:
            assert(imbalance is not None)
            for ratio in imbalance:
                c_criterion.append(torch.nn.BCELoss(weight=torch.FloatTensor([ratio])).to(device)) # Note this was originally BCEwithLogitsLoss, but I change the output to sigmoid for consistency over all models.
        else:
            for i in range(args.n_attributes):
                c_criterion.append(torch.nn.CrossEntropyLoss().to(device))
    else:
        c_criterion = None

    #Define the optimizer
    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'RMSprop':
        optimizer = torch.optim.RMSprop(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)


    #Define the learning rate scheduler Note: the commented out scheduler was commented out in the orignal code #TODO check if this is correct
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=5, threshold=0.00001, min_lr=0.00001, eps=1e-08)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.scheduler_step, gamma=0.1)
    stop_epoch = int(math.log(args.min_lr / args.lr) / math.log(args.lr_decay_size)) * args.scheduler_step
    print("Stop epoch: ", stop_epoch)


    #Define dataloaders
    train_data_path = os.path.join(args.data_dir, 'train.pkl')
    val_data_path = train_data_path.replace('train.pkl', 'val.pkl')
    logger.write('train data path: %s\n' % train_data_path)

    train_data_path = os.path.join(args.data_dir, 'train.pkl')
    val_data_path = train_data_path.replace('train.pkl', 'val.pkl')

    #Set the data loaders to not load images if the model does not require them
    if train_mode == "C_to_Y":
        no_img = True
    else:
        no_img = False

    # Set the use_attr flag to false if the model does not require attributes
    if train_mode == "X_to_Y":
        use_attr = False
    else:
        use_attr = True

    if args.ckpt: #retraining
        train_loader = load_data([train_data_path, val_data_path], use_attr=use_attr, no_img=no_img, batch_size=args.batch_size, uncertain_label=args.uncertain_labels, image_dir=args.image_dir, n_class_attr=args.n_class_attr, resampling=args.resampling)
        val_loader = None
    else:
        train_loader = load_data([train_data_path],  use_attr=use_attr, no_img=no_img, batch_size=args.batch_size, uncertain_label=args.uncertain_labels, image_dir=args.image_dir, n_class_attr=args.n_class_attr, resampling=args.resampling)
        val_loader = load_data([val_data_path],  use_attr=use_attr, no_img=no_img, batch_size=args.batch_size, uncertain_label=args.uncertain_labels, image_dir=args.image_dir, n_class_attr=args.n_class_attr, resampling=args.resampling)

    best_val_epoch = -1
    best_val_loss = float('inf')
    best_val_acc = 0

    for epoch in range(0, args.epochs):

        if train_mode == "X_to_Y":
            train_loss_meter, train_acc_meter = run_epoch_XtoY(model, optimizer, train_loader, y_criterion, args,device, is_training=True)

        elif train_mode == "C_to_Y":
            train_loss_meter, train_acc_meter = run_epoch_CtoY(model, optimizer, train_loader, y_criterion, args,device, is_training=True)

        elif train_mode == "X_to_C":
            train_loss_meter, train_acc_meter = run_epoch_XtoC(model, optimizer, train_loader, c_criterion, args,device, is_training=True)

        elif train_mode == "X_to_C_to_Y":
            train_loss_meter,train_c_loss_meter,train_y_loss_meter, train_c_acc_meter,y_acc_meter = run_epoch_XtoCtoY(model, optimizer, train_loader, y_criterion, c_criterion, args,device, is_training=True)
        
        if not args.ckpt: # evaluate on val set
            with torch.no_grad():
                if train_mode == "X_to_Y":
                    val_loss_meter, val_acc_meter = run_epoch_XtoY(model, optimizer, val_loader, y_criterion, args,device, is_training=False)

                elif train_mode == "C_to_Y":
                    val_loss_meter, val_acc_meter = run_epoch_CtoY(model, optimizer, val_loader, y_criterion, args,device, is_training=False)

                elif train_mode == "X_to_C":
                    val_loss_meter, val_acc_meter = run_epoch_XtoC(model, optimizer, val_loader, c_criterion, args,device, is_training=False)

                elif train_mode == "X_to_C_to_Y":
                    train_loss_meter,val_c_loss_meter,val_y_loss_meter, val_c_acc_meter,val_y_acc_meter = run_epoch_XtoCtoY(model, optimizer, val_loader,  y_criterion, c_criterion, args,device, is_training=False)
                    
                    # The original paper use the y accuracy meter for evaluation. However, a weighted average of the two accuracies could be used.
                    val_acc_meter = val_y_acc_meter 
                    val_loss_meter=val_y_loss_meter
        
        else: #retraining
            val_loss_meter = train_loss_meter
            val_acc_meter = train_acc_meter


        if best_val_acc < val_acc_meter.avg:
            best_val_epoch = epoch
            best_val_acc = val_acc_meter.avg
            logger.write('New model best model at epoch %d\n' % epoch)
            torch.save(model, os.path.join(args.log_dir, 'best_model_%d.pth' % args.seed))
            #if best_val_acc >= 100: #in the case of retraining, stop when the model reaches 100% accuracy on both train + val sets
            #    break

        train_loss_avg = train_loss_meter.avg
        val_loss_avg = val_loss_meter.avg

        if train_mode == "X_to_C_to_Y":
            # Print the losses and accuracies for the concept and class prediction
            train_logger.write(f"Epoch [{epoch:2d}]: "
                            f"Train loss: {train_loss_avg:.4f} "
                            f"Train class loss: {train_c_loss_meter.avg:.4f} "
                            f"Train label loss: {train_y_loss_meter.avg:.4f} "
                            f"Train class accuracy: {train_c_acc_meter.avg:.4f} "
                            f"Train label accuracy: {y_acc_meter.avg:.4f} "
                            f"Val class loss: {val_c_loss_meter.avg:.4f} "
                            f"Val label loss: {val_y_loss_meter.avg:.4f} "
                            f"Val class accuracy: {val_c_acc_meter.avg:.4f} "
                            f"Val label accuracy: {val_y_acc_meter.avg:.4f} "
                            f"Val loss: {val_loss_avg:.4f} "
                            f"Best val epoch: {best_val_epoch}\n")
            
        else:
            train_logger.write(f"Epoch [{epoch:2d}]: "
                            f"Train loss: {train_loss_avg:.4f} "
                            f"Train accuracy: {train_acc_meter.avg:.4f} "
                            f"Val loss: {val_loss_avg:.4f} "
                            f"Val acc: {val_acc_meter.avg:.4f} "
                            f"Best val epoch: {best_val_epoch}\n")
        

        train_logger.flush()
  
        if epoch <= stop_epoch:
            scheduler.step(epoch) #scheduler step to update lr at the end of epoch     
        #inspect lr
        if epoch % 10 == 0:
            print('Current lr:', scheduler.get_lr())

        # if epoch % args.save_step == 0:
        #     torch.save(model, os.path.join(args.log_dir, '%d_model.pth' % epoch))

        if epoch >= 100 and val_acc_meter.avg < 3:
            print("Early stopping because of low accuracy")
            break
        if epoch - best_val_epoch >= 100:
            print("Early stopping because acc hasn't improved for a long time")
            break

def train_X_to_C(args):
    """
    Train concept prediction model used in independent and sequential training
    """
    train_mode = "X_to_C"
    model = ModelXtoC(pretrained=args.pretrained, freeze=args.freeze, use_aux=args.use_aux,
                      n_attributes=args.n_attributes)
    train(model,train_mode, args)

def train_C_to_Y(args):
    train_mode = "C_to_Y"
    model = ModelCtoY(n_attributes=args.n_attributes,
                            num_classes=args.n_classes)
    train(model,train_mode, args)


def train_X_to_C_to_y(args):

    train_mode = "X_to_C_to_Y"
    model = ModelXtoCtoY(pretrained=args.pretrained, freeze=args.freeze,
                         num_classes=args.n_classes, use_aux=args.use_aux, n_attributes=args.n_attributes)
    train(model,train_mode, args)

def train_X_to_y(args):
    train_mode = "X_to_Y"
    model = ModelXtoY(pretrained=args.pretrained, freeze=args.freeze, num_classes=args.n_classes, use_aux=args.use_aux)
    train(model,train_mode, args)
