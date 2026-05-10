import torch
import torch.nn as nn
import torch.nn.functional as F
from core.update import BasicMultiUpdateBlock
from core.extractor import BasicEncoder, MultiBasicEncoder, ResidualBlock
from core.corr import CorrBlock1D, PytorchAlternateCorrBlock1D
from core.utils.utils import coords_grid, updisp8

try:
    autocast = torch.cuda.amp.autocast
except:
    # dummy autocast for PyTorch < 1.6
    class autocast:
        def __init__(self, enabled):
            pass
        def __enter__(self):
            pass
        def __exit__(self, *args):
            pass

class SpikeStereoNet(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        
        context_dims = args.hidden_dims   # default: [128]*3, (list: [128, 128, 128])

        self.cnet = MultiBasicEncoder(output_dim=[args.hidden_dims, context_dims], norm_fn=args.context_norm, downsample=args.n_downsample)  
        
        self.update_block = BasicMultiUpdateBlock(self.args, hidden_dims=args.hidden_dims)

        self.context_zqr_convs = nn.ModuleList([nn.Conv2d(context_dims[i], args.hidden_dims[i]*3, 3, padding=3//2) for i in range(self.args.n_snn_layers)])

        if args.shared_backbone:
            self.conv2 = nn.Sequential(
                ResidualBlock(128, 128, 'instance', stride=1),
                nn.Conv2d(128, 256, 3, padding=1))
        else:
            self.fnet = BasicEncoder(output_dim=256, norm_fn='instance', downsample=args.n_downsample)

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def initialize_disp(self, img):
        """ Disparity is represented as difference between two coordinate grids disparity = coords1 - coords0"""
        N, _, H, W = img.shape

        coords0 = coords_grid(N, H, W).to(img.device)   # [batch, 2, ht, wd]
        coords1 = coords_grid(N, H, W).to(img.device)

        return coords0, coords1

    def upsample_disp(self, disp, mask):
        """ Upsample disparity field [H/8, W/8, 2] -> [H, W, 2] using convex combination """
        N, D, H, W = disp.shape   # [B, 1, H, W]
        factor = 2 ** self.args.n_downsample    # default: n_downsample=2
        mask = mask.view(N, 1, 9, factor, factor, H, W)
        mask = torch.softmax(mask, dim=2)

        up_disp = F.unfold(factor * disp, [3,3], padding=1)
        up_disp = up_disp.view(N, D, 9, 1, 1, H, W)

        up_disp = torch.sum(mask * up_disp, dim=2)
        up_disp = up_disp.permute(0, 1, 4, 2, 5, 3)
        return up_disp.reshape(N, D, factor*H, factor*W)


    def forward(self, spike1, spike2, iters=16, disp_init=None, test_mode=False):
        """ Estimate disparity between left and right spike pairs """

        # spike1 = (2 * (spike1 / 255.0) - 1.0).contiguous()
        # spike2 = (2 * (spike2 / 255.0) - 1.0).contiguous()

        spike1 = (2 * spike1 - 1.0).contiguous()
        spike2 = (2 * spike2 - 1.0).contiguous()

        # run the context network
        with autocast(enabled=self.args.mixed_precision):
            if self.args.shared_backbone:
                *cnet_list, x = self.cnet(torch.cat((spike1, spike2), dim=0), dual_inp=True, num_layers=self.args.n_snn_layers)
                fmap1, fmap2 = self.conv2(x).split(dim=0, split_size=x.shape[0]//2)
            else:
                cnet_list = self.cnet(spike1, num_layers=self.args.n_snn_layers)
                fmap1, fmap2 = self.fnet([spike1, spike2])

            net_list = [torch.tanh(x[0]) for x in cnet_list]
            inp_list = [torch.relu(x[1]) for x in cnet_list]

            inp_list = [list(conv(i).split(split_size=conv.out_channels//3, dim=1)) for i,conv in zip(inp_list, self.context_zqr_convs)]        


        if self.args.corr_implementation == "reg": # Default
            corr_block = CorrBlock1D
            fmap1, fmap2 = fmap1.float(), fmap2.float()
        elif self.args.corr_implementation == "alt": # More memory efficient than reg
            corr_block = PytorchAlternateCorrBlock1D
            fmap1, fmap2 = fmap1.float(), fmap2.float()
        corr_fn = corr_block(fmap1, fmap2, radius=self.args.corr_radius, num_levels=self.args.corr_levels)

        coords0, coords1 = self.initialize_disp(net_list[0])

        if disp_init is not None:
            coords1 = coords1 + disp_init

        disp_predictions = []
        
        s_list = []
        v_list = []
        
        for itr in range(iters):    # train: 16
            coords1 = coords1.detach()  # Detach to prevent gradients from flowing into corr_fn, which is not a PyTorch module and does not support autograd
            corr = corr_fn(coords1) # index correlation volume
            disp = coords1 - coords0
            with autocast(enabled=self.args.mixed_precision):
                net_list, v, up_mask, delta_disp = self.update_block(net_list, inp_list, corr, disp, iter32=self.args.n_snn_layers==3, iter16=self.args.n_snn_layers>=2)

            # in stereo mode, project disparity onto epipolar
            delta_disp[:,1] = 0.0

            # F(t+1) = F(t) + \Delta(t)
            coords1 = coords1 + delta_disp
            
            s_list.append(net_list[0])
            v_list.append(v)

            # We do not need to upsample or output intermediate results in test_mode
            if test_mode and itr < iters-1:
                continue

            # upsample predictions
            if up_mask is None:  
                disp_up = updisp8(coords1 - coords0)
            else:
                disp_up = self.upsample_disp(coords1 - coords0, up_mask)

            disp_up = disp_up[:,:1]

            disp_predictions.append(disp_up)

        
        if test_mode:
            return coords1 - coords0, disp_up, s_list
        
        return disp_predictions, s_list, v_list
