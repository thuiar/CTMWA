import torch
import os
from models.base import BaseModel
from models.networks import concatFusion
import torch.nn as nn
import torch.nn.functional as F
from .tools import Linear_fw
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
from models.networks import seqEncoder
from transformers import BertModel
import numpy as np
import time
from torch import exp

class BlockUnit(nn.Module):
    def __init__(self, in_dim, mid_dim, out_dim) -> None:
        super().__init__()
        self.linear1 = Linear_fw(in_dim, mid_dim)
        self.linear2 = Linear_fw(mid_dim, out_dim)

    def forward(self, x):
        out = self.linear1(x)
        out = F.leaky_relu(out, negative_slope=0.1, inplace=False)
        out = self.linear2(out)
        return out
    
class WUnit(nn.Module):
    def __init__(self, in_dim, mid_dim, out_dim) -> None:
        super().__init__()
        self.linear1 = nn.Linear(in_dim, mid_dim)
        self.relu = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(mid_dim, out_dim)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        out = self.linear2(x)
        return torch.sigmoid(out)
    
class Ctmwa(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        # Model Parameters
        parser.add_argument('--input_dim_t', type=int, default=512, help='lexical input dim')
        parser.add_argument('--input_dim_v', type=int, default=512, help='visual input dim')

        parser.add_argument('--embd_size', default=128, type=int, help='model embedding size')

        parser.add_argument('--fusion_method', default='concat', type=str, choices=['concat', 'add', 'mul', 'tensor'])
        parser.add_argument('--cls_layers', type=str, default='64,64', help='256,128 for 2 layers with 256, 128 nodes respectively')
        parser.add_argument('--output_dim', type=int, default=7, help='output classification. linear classification')
        parser.add_argument('--dropout_rate', type=float, default=0.3, help='rate of dropout')
        
        # Training Parameters.
        parser.add_argument('--batch_size', type=int, default=32, help='input batch size')
        
        parser.add_argument('--niter', type=int, default=12, help='# of iter at starting learning rate')
        parser.add_argument('--niter_decay', type=int, default=12, help='# of iter to linearly decay learning rate to zero')
        parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        parser.add_argument('--num_heads', type=int, default=3, help='# of iter at starting learning rate')

        parser.add_argument('--trans_tit', type=float, default=1.0, help='initial learning rate for adam')
        parser.add_argument('--trans_iti', type=float, default=0.05, help='initial learning rate for adam')
        parser.add_argument('--trans_ti', type=float, default=4.0, help='initial learning rate for adam')
        parser.add_argument('--trans_it', type=float, default=0.1, help='initial learning rate for adam')
        
        parser.add_argument('--lr', type=float, default=0.0005, help='initial learning rate for adam')
        parser.add_argument('--inner_lr', type=float, default=0.005, help='initial learning rate for adam')
        parser.add_argument('--wnet_lr', type=float, default=0.0005, help='initial learning rate for adam')
        
        parser.add_argument('--lr_policy', type=str, default='linear', help='learning rate policy. [linear | step | plateau | cosine]')
        parser.add_argument('--weight_decay', type=float, default=2e-4, help='weight decay when training')

        return parser

    def __init__(self, opt):
        """
        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        super().__init__(opt)
        self.loss_names = ['cls']
        self.model_names = ['txt_encoder','img2txt', 'txt2img', 'txt_neck',
        'img_neck','net_fusion','net_m_c','net_t_c','net_v_c','txt_w_net', 'img_w_net']

        self.backbone_names = ['txt_encoder','img2txt', 'txt2img', 'txt_neck',
        'img_neck','net_fusion','net_m_c','net_t_c','net_v_c']
        
        self.wnet_models = ['txt_w_net', 'img_w_net']

        self.txt_encoder = Linear_fw(opt.input_dim_t, opt.input_dim_t)

        self.txt_w_net = WUnit(in_dim = opt.output_dim*3+opt.input_dim_t+opt.input_dim_v, mid_dim = 128, out_dim = 1)
        self.img_w_net = WUnit(in_dim = opt.output_dim*3+opt.input_dim_t+opt.input_dim_v, mid_dim = 128, out_dim = 1)

        self.img2txt = seqEncoder(opt.input_dim_v, opt.input_dim_t, dropout=opt.dropout_rate)
        self.txt2img = seqEncoder(opt.input_dim_t, opt.input_dim_v, dropout=opt.dropout_rate)

        self.txt_neck = BlockUnit(in_dim = opt.input_dim_v + opt.input_dim_t, mid_dim = int((opt.input_dim_t+opt.input_dim_v+opt.embd_size)/2), out_dim = opt.embd_size)
        self.img_neck = BlockUnit(in_dim = opt.input_dim_v + opt.input_dim_t, mid_dim = int((opt.input_dim_t+opt.input_dim_v+opt.embd_size)/2), out_dim = opt.embd_size)
        
        # self.net_fusion = MultiHeadAttention(opt.embd_size*2, opt.embd_size, num_heads=opt.num_heads)
        self.net_fusion = concatFusion(opt)
        # opt.cls_input_size = (opt.embd_size+ 1) ** 2

        self.fast_parameters = []
        self.net_m_c = BlockUnit(in_dim = opt.embd_size*2 , mid_dim = int((opt.embd_size*2 + opt.output_dim)/2), out_dim = opt.output_dim)
        self.net_t_c = BlockUnit(in_dim = opt.embd_size , mid_dim = int((opt.embd_size + opt.output_dim)/2), out_dim = opt.output_dim)
        self.net_v_c = BlockUnit(in_dim = opt.embd_size , mid_dim = int((opt.embd_size + opt.output_dim)/2), out_dim = opt.output_dim)

        if self.isTrain:
            self.criterion_ce = opt.criterion_clsloss
            self.criterion_l1 = nn.L1Loss()

            self.wnet_paremeter = [{'params': getattr(self, net).parameters()} for net in self.wnet_models]
            self.wnet_optimizer = torch.optim.Adam(self.wnet_paremeter, lr=opt.wnet_lr, betas=(opt.beta1, 0.999), weight_decay=opt.weight_decay)
            self.optimizers.append(self.wnet_optimizer)

            self.paremeter = [{'params': getattr(self, net).parameters()} for net in self.backbone_names]
            self.model_optimizer = torch.optim.SGD(self.paremeter, lr=opt.lr, momentum=0.9, weight_decay=opt.weight_decay)
            self.optimizers.append(self.model_optimizer)

            self.output_dim = opt.output_dim

        # modify save_dir
        if not os.path.exists(self.save_dir) and opt.save_model:
            os.mkdir(self.save_dir)
        self.save_dir = os.path.join(self.save_dir, str(opt.seed))
        if not os.path.exists(self.save_dir) and opt.save_model:
            os.mkdir(self.save_dir)
    
    def update_params(self, grad):
        self.fast_parameters = []
        pointer = 0
        for m in self.backbone_names:
            for weight in getattr(self, m).parameters():
                if weight.fast is None:
                    weight.fast = weight - self.opt.inner_lr * grad[pointer] # create weight.fast 
                else:
                    weight.fast = weight.fast - self.opt.inner_lr * grad[pointer] # update weight.fast
                pointer += 1
                self.fast_parameters.append(weight.fast) # gradients are based on newest weights, but the graph will retain the link to old weight.fasts

    def inner_support_train(self, support_batch, epoch):
        text = support_batch['text'].float().to(self.device)
        image = support_batch['image'].float().to(self.device)
        label = support_batch['label'].type(torch.int64).to(self.device)

        output_m, output_t, output_v, text, image, txt_img_txt, img_txt_img, img_txt, txt_img, fusion_t, fusion_v = self.forward(text, image)

        supprot_losses = self.criterion_l1(text, txt_img_txt) * self.opt.trans_tit
        supprot_losses += self.criterion_l1(image, img_txt_img) * self.opt.trans_iti

        supprot_losses += self.criterion_l1(text, img_txt) * self.opt.trans_it
        supprot_losses += self.criterion_l1(image, txt_img) * self.opt.trans_ti

        cost_m = F.cross_entropy(output_m, label.long(), reduction='none')
        cost_m = torch.reshape(cost_m, (len(cost_m), 1))

        cost_t = F.cross_entropy(output_t, label.long(), reduction='none')
        cost_t = torch.reshape(cost_t, (len(cost_t), 1))

        cost_v = F.cross_entropy(output_v, label.long(), reduction='none')
        cost_v = torch.reshape(cost_v, (len(cost_v), 1))
        
        cost_w_t = torch.cat((torch.nn.functional.one_hot(label.long(),num_classes=self.opt.output_dim),output_m,output_t,fusion_t),-1)
        cost_w_v = torch.cat((torch.nn.functional.one_hot(label.long(),num_classes=self.opt.output_dim),output_m,output_v,fusion_v),-1)

        txt_w_lambda = self.txt_w_net(cost_w_t.data)
        txt_w_meta = torch.sum(cost_t * txt_w_lambda)/len(cost_t)

        img_w_lambda = self.img_w_net(cost_w_v.data)
        img_w_meta = torch.sum(cost_v * img_w_lambda)/len(cost_v)

        m_w_meta = torch.sum(cost_m)/len(cost_m)

        supprot_losses += txt_w_meta+img_w_meta+m_w_meta

        self.zero_grad()

        grad = torch.autograd.grad(supprot_losses, self.fast_parameters, create_graph=True)

        self.update_params(grad)

        del grad

    def inner_query_train(self, epoch):
        output_m, output_t, output_v, _,_, _, _, _, _, _, _ = self.forward(self.text, self.image)

        m_meta = F.cross_entropy(output_m, self.label.long())
        t_meta = F.cross_entropy(output_t, self.t_label.long())
        v_meta = F.cross_entropy(output_v, self.v_label.long())
        query_losses = m_meta + t_meta + v_meta

        self.wnet_zero_grad()
        query_losses.backward()
        self.wnet_optimizer.step()
    
    def inner_train(self, tr_loader, meta_loader, logger, epoch):
        tr_losses = 0
        preds, labels = [], []
        meta_loader_iter = iter(meta_loader)
        epoch_start_time = time.time()
        for batch_data in tqdm(tr_loader):
            self.net_reset()
            self.inner_support_train(batch_data, epoch)
            try:
                meta_batch = next(meta_loader_iter)
            except StopIteration:
                meta_loader_iter = iter(meta_loader)
                meta_batch = next(meta_loader_iter)

            self.set_input(meta_batch)
            self.inner_query_train(epoch)

            self.set_input(batch_data)
            self.net_reset()
            output_m, output_t, output_v, text, image, txt_img_txt, img_txt_img, img_txt, txt_img, fusion_t, fusion_v = self.forward(self.text, self.image)
            
            losses = self.criterion_l1(text, txt_img_txt) * self.opt.trans_tit
            losses += self.criterion_l1(image, img_txt_img) * self.opt.trans_iti

            losses += self.criterion_l1(text, img_txt) * self.opt.trans_it
            losses += self.criterion_l1(image, txt_img) * self.opt.trans_ti

            cost_m = F.cross_entropy(output_m, self.label.long(), reduction='none')
            cost_m = torch.reshape(cost_m, (len(cost_m), 1))

            cost_t = F.cross_entropy(output_t, self.label.long(), reduction='none')
            cost_t = torch.reshape(cost_t, (len(cost_t), 1))

            cost_v = F.cross_entropy(output_v, self.label.long(), reduction='none')
            cost_v = torch.reshape(cost_v, (len(cost_v), 1))

            cost_w_t = torch.cat((torch.nn.functional.one_hot(self.label.long(),num_classes=self.opt.output_dim),output_m,output_t,fusion_t),-1)
            cost_w_v = torch.cat((torch.nn.functional.one_hot(self.label.long(),num_classes=self.opt.output_dim),output_m,output_v,fusion_v),-1)

            with torch.no_grad():
                txt_w_new = self.txt_w_net(cost_w_t)
                img_w_new = self.img_w_net(cost_w_v)

            txt_w_meta = torch.sum(cost_t * txt_w_new)/len(cost_t)
            img_w_meta = torch.sum(cost_v * img_w_new)/len(cost_v)
            m_w_meta = torch.sum(cost_m)/len(cost_m)

            losses += txt_w_meta + img_w_meta + m_w_meta

            tr_losses += losses.item()
            
            labels.append(self.label.cpu().detach().numpy())
            preds.append(output_m.cpu().detach().numpy())

            self.zero_grad()
            self.backward(losses)
            self.model_optimizer.step()
        
        labels = np.concatenate(labels, axis=0)
        preds = np.concatenate(preds, axis=0)
        tr_losses = tr_losses / len(tr_loader)

        logger.info(f'End of training epoch {epoch} / {self.opt.niter + self.opt.niter_decay} \t Time Taken: {time.time() - epoch_start_time} sec')
        preds = np.argmax(preds, axis=1)
        accuracy = accuracy_score(labels.flatten(), preds.flatten())
        f1 = f1_score(labels.flatten(), preds.flatten(), average='weighted')
        logger.info(f'Train Loss: {tr_losses} \t Acc: {accuracy} \t F1: {f1}')

    def inner_test(self, ts_loader, logger, type):
        self.fusion_m = np.array([])
        preds, labels = [], []
        ids = []
        ts_losses = 0
        for batch_data in tqdm(ts_loader):
            self.set_input(batch_data)

            output_m, _, _, _, _, _, _,_,_,_,_ = self.forward(self.text, self.image)

            ts_cost = F.cross_entropy(output_m, self.label.long())
            ts_losses += ts_cost.item()
            labels.append(self.label.cpu().detach().numpy())
            preds.append(output_m.cpu().detach().numpy())
            ids.append(np.array(batch_data['id']))

        labels = np.concatenate(labels, axis=0)
        preds = np.concatenate(preds, axis=0)
        ids = np.concatenate(ids, axis=0)

        ts_losses = ts_losses / len(ts_loader)
        preds = np.argmax(preds, axis=1)
        accuracy = accuracy_score(labels.flatten(), preds.flatten())
        f1 = f1_score(labels.flatten(), preds.flatten(), average='weighted')
        logger.info(f'{type} Loss: {ts_losses} \t Acc: {accuracy} \t F1: {f1}')

        return accuracy, f1, ids, labels, preds

    def net_reset(self):
        self.fast_parameters = self.get_inner_loop_params()
        for m in self.backbone_names:
            for weight in getattr(self, m).parameters():  # reset fast parameters
                weight.fast = None

    def get_inner_loop_params(self):
        inner_loop_p = []
        for m in self.backbone_names:
            inner_loop_p.extend(list(getattr(self, m).parameters()))
        return inner_loop_p

    def forward(self,text, image):
        text = self.txt_encoder(text)

        txt_img = self.txt2img(text)
        txt_img_txt = self.img2txt(txt_img)

        img_txt = self.img2txt(image)
        img_txt_img = self.txt2img(img_txt)

        fusion_t = torch.cat((text, txt_img), 1)
        fusion_v = torch.cat((img_txt, image), 1)

        mlp_t = self.txt_neck(fusion_t)
        mlp_v = self.img_neck(fusion_v)

        fusion_m = self.net_fusion(mlp_t, mlp_v)

        output_m = self.net_m_c(fusion_m)
        output_t = self.net_t_c(mlp_t)
        output_v = self.net_v_c(mlp_v)

        return output_m, output_t, output_v, text,image, txt_img_txt, img_txt_img, img_txt, txt_img, fusion_t, fusion_v

    def backward(self,losses):
        losses.backward()
        for m in self.backbone_names:
            torch.nn.utils.clip_grad_norm_(getattr(self, m).parameters(), 2)
    
    def set_input(self, input):
        self.text = input['text'].float().to(self.device)
        self.image = input['image'].float().to(self.device)
        self.label = input['label'].type(torch.int64).to(self.device)
        if 't_label' in input:
            self.t_label = input['t_label'].type(torch.int64).to(self.device)
            self.v_label = input['v_label'].type(torch.int64).to(self.device)

    def optimize_parameters(self):
        self.forward()
        self.zero_grad()
        self.backward()
        self.model_optimizer.step()

    def zero_grad(self):
        for m in self.backbone_names:
            getattr(self, m).zero_grad()
    
    def wnet_zero_grad(self):
        for m in self.wnet_models:
            getattr(self, m).zero_grad()
