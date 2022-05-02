import sys, os
sys.path.append('./CenterNet/src/lib/')

from opts import opts
from models.model import create_model, load_model
import torch
from torch.onnx import OperatorExportTypes
from collections import OrderedDict
import argparse

def torch2onnx(torch_ckpt, onnx_ckpt):
    opt = opts().init()
    opt.arch = 'dla_34'
    opt.task = 'ctdet'
    opt.heads = OrderedDict([('hm', 80), ('reg', 2), ('wh', 2)])
    opt.head_conv = 256 if 'dla' in opt.arch else 64
    model = create_model(opt.arch, opt.heads, opt.head_conv)
    load_model(model, torch_ckpt)
    model.eval()
    model.cuda()
    input = torch.rand((1, 3, 512, 512), dtype=torch.float32).cuda()
    torch.onnx.export(model, input, onnx_ckpt, verbose=True,
                    operator_export_type=OperatorExportTypes.ONNX, 
                    enable_onnx_checker=True, output_names=['hm', 'wh', 'reg'])
    print(f'Converted {torch_ckpt} to {onnx_ckpt}')

def onnx2trt(onnx_ckpt, trt_ckpt):
    os.system(f'./onnx-tensorrt/build/onnx2trt {onnx_ckpt} -o {trt_ckpt}')
    print(f'Converted {onnx_ckpt} to {trt_ckpt}')

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--torch-ckpt', type=str, default='weights/ctdet_coco_dla_2x.pth')
    parser.add_argument('--onnx-ckpt', type=str, default='weights/ctdet_coco_dla_2x.onnx')
    parser.add_argument('--trt-ckpt', type=str, default='weights/ctdet_coco_dla_2x.trt')
    args = parser.parse_args()
    torch2onnx(torch_ckpt=args.torch_ckpt, onnx_ckpt=args.onnx_ckpt)
    onnx2trt(onnx_ckpt=args.onnx_ckpt, trt_ckpt=args.trt_ckpt)
