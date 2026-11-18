import numpy as np
import torch
import gradio as gr
from PIL import Image
from net.CIDNet import CIDNet
import torchvision.transforms as transforms
import torch.nn.functional as F
import os
import argparse
import time

opt_parser = argparse.ArgumentParser(description='App')
opt_parser.add_argument('--cpu', action='store_true', help='CPU-Only')
opt_parser.add_argument('--port', type=int, default=7862, help='Local web port')
opt = opt_parser.parse_args()

if opt.cpu:
    eval_net = CIDNet().cpu()
else:
    eval_net = CIDNet().cuda()
    
eval_net.trans.gated = True
eval_net.trans.gated2 = True

def brisque_score_compatible(image):
    """Keep upstream BRISQUE features, adapting old skimage/libsvm APIs."""
    import imquality.brisque as brisque
    from skimage.transform import rescale

    full = brisque.Brisque(image, kernel_size=7, sigma=7 / 6)
    half_gray = rescale(full.image, 0.5, order=2, mode='constant',
                        anti_aliasing=False, channel_axis=None)
    # Brisque expects RGB even when its input is already grayscale.
    half = brisque.Brisque(np.repeat(half_gray[..., None], 3, axis=-1),
                           kernel_size=7, sigma=7 / 6)
    features = brisque.scale_features(np.concatenate([full.features, half.features]))
    # Legacy libsvm's numpy adapter references removed scipy.ndarray aliases.
    nodes = (brisque.svmutil.svm_node * (len(features) + 1))()
    for index, value in enumerate(features):
        nodes[index].index = index + 1
        nodes[index].value = float(value)
    nodes[len(features)].index = -1
    estimates = (brisque.svmutil.c_double * 1)()
    return brisque.svmutil.libsvm.svm_predict_probability(brisque.model, nodes, estimates)

def process_image(input_img,score,model_path,gamma,alpha_s=1.0,alpha_i=1.0,
                  max_side=768, progress=gr.Progress()):
    if input_img is None:
        raise gr.Error("Please choose an input image.")
    if not model_path:
        raise gr.Error("Please choose model weights from the list.")
    started = time.perf_counter()
    original_size = input_img.size
    input_img = input_img.convert('RGB')
    if str(max_side) != 'Original':
        input_img = input_img.copy()
        limit = int(max_side)
        input_img.thumbnail((limit, limit), Image.Resampling.LANCZOS)
    elif opt.cpu and input_img.width * input_img.height > 2_000_000:
        gr.Warning("Full-resolution CPU processing may take several minutes. Choose 768 for a quick preview.")
    processed_size = input_img.size
    progress(0.1, desc=f"Preparing {processed_size[0]} x {processed_size[1]} image")
    torch.set_grad_enabled(False)
    eval_net.load_state_dict(torch.load(os.path.join(directory,model_path), map_location='cpu', weights_only=True))
    eval_net.eval()
    
    pil2tensor = transforms.Compose([transforms.ToTensor()])
    input = pil2tensor(input_img)
    factor = 8
    h, w = input.shape[1], input.shape[2]
    H, W = ((h + factor) // factor) * factor, ((w + factor) // factor) * factor
    padh = H - h if h % factor != 0 else 0
    padw = W - w if w % factor != 0 else 0
    input = F.pad(input.unsqueeze(0), (0,padw,0,padh), 'reflect')
    progress(0.2, desc="Enhancing image")
    with torch.no_grad():
        eval_net.trans.alpha_s = alpha_s
        eval_net.trans.alpha = alpha_i
        if opt.cpu:
            output = eval_net(input**gamma)
        else:
            output = eval_net(input.cuda()**gamma)
            
    if opt.cpu:
        output = torch.clamp(output,0,1)
    else:
        output = torch.clamp(output.cuda(),0,1).cuda()
    output = output[:, :, :h, :w]
    enhanced_img = transforms.ToPILImage()(output.squeeze(0))
    progress(0.85, desc="Preparing result")
    score_niqe = score_brisque = None
    if score == 'Yes':
        progress(0.9, desc="Calculating image quality scores")
        from loss.niqe_utils import calculate_niqe
        im1 = enhanced_img.convert('RGB')
        score_brisque = brisque_score_compatible(im1)
        im1 = np.array(im1)
        score_niqe = calculate_niqe(im1)
    elapsed = time.perf_counter() - started
    details = (f"Input: {original_size[0]} x {original_size[1]} | "
               f"Output: {processed_size[0]} x {processed_size[1]} | "
               f"{'CPU' if opt.cpu else 'CUDA'} | {elapsed:.2f} seconds")
    if processed_size != original_size:
        details += " | Resized preview; select Original for full-resolution evaluation."
    print(details, flush=True)
    progress(1.0, desc="Done")
    return enhanced_img,score_niqe,score_brisque,details

def find_pth_files(directory):
    pth_files = []
    for root, dirs, files in os.walk(directory):
        if 'train' in root.split(os.sep):
            continue
        for file in files:
            if file.endswith('.pth'):
                pth_files.append(os.path.join(root, file))
    return pth_files

def remove_weights_prefix(paths):
    return sorted(os.path.relpath(path, directory) for path in paths)

directory = "weights"
pth_files = find_pth_files(directory)
pth_files2 = remove_weights_prefix(pth_files)

interface = gr.Interface(
    fn=process_image,
    inputs=[
        gr.Image(label="Low-light Image", type="pil"),
        gr.Radio(choices=['Yes','No'],value='No',label="Image Score",info="Calculate NIQE and BRISQUE, default is \"No\"."),
        gr.Radio(choices=pth_files2,value=pth_files2[0] if pth_files2 else None,label="Model Weights",info="Choose the checkpoint matching your dataset."),
        gr.Slider(0.1,5,label="gamma curve",step=0.01,value=1.0, info="Lower is lighter, and best range is [0.5,2.5]."),
        gr.Slider(0,2,label="Alpha-s",step=0.01,value=1.0, info="Higher is more saturated."),
        gr.Slider(0.1,2,label="Alpha-i",step=0.01,value=1.0, info="Higher is lighter."),
        gr.Dropdown(choices=['512', '768', '1024', 'Original'], value='768',
                    label="Processing size (long edge)",
                    info="768 is a fast preview. Output is resized. Original keeps full resolution and can be slow on CPU.")
    ],
    outputs=[
        gr.Image(label="Result", type="pil", format='png'),
        gr.Textbox(label="NIQE",info="Lower is better."),
        gr.Textbox(label="BRISQUE",info="Lower is better."),
        gr.Textbox(label="Processing details")
    ],
    title="HVI-CIDNet (Low-Light Image Enhancement)",
    allow_flagging="never"
)

if __name__ == '__main__':
    interface.launch(server_name='127.0.0.1', server_port=opt.port, share=False)
