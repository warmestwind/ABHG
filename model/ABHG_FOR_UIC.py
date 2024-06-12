from .UNet_S import Unet

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np



num_pt = 4
num_patch = num_pt * 9
num_hyper = 16
patch_size = 8



def getcropedInputs(ROIs, inputs_origin, cropSize, useGPU=0):
    landmarks = ROIs.detach().cpu().numpy()
    landmarkNum = landmarks.shape[1]
    b, c, h, w = inputs_origin.size()

    cropSize = int(cropSize / 2)

    X, Y = landmarks[:, :, 0], landmarks[:, :, 1] + 1

    X, Y = np.round(X * (h - 1)).astype("int"), np.round(Y * (w - 1)).astype("int")

    cropedDICOMs = []
    flag = True
    for landmarkId in range(landmarkNum):
        x, y = X[:, landmarkId].clip(0, 255), Y[:, landmarkId].clip(0, 255)
        lx, ux, ly, uy = x - cropSize, x + cropSize, y - cropSize, y + cropSize
        lxx, uxx, lyy, uyy = np.where(lx > 0, lx, 0), np.where(ux < h, ux, h), np.where(ly > 0, ly, 0), np.where(uy < w,
                                                                                                                 uy, w)
        # lxx, uxx, lyy, uyy = np.clip(lx, 0, 255), np.clip(ux, 0, 255), np.clip(ly, 0, 255), np.clip(uy, 0, 255)
        for b_id in range(b):
            cropedDICOM = inputs_origin[b_id:b_id + 1, :, lxx[b_id]: uxx[b_id], lyy[b_id]: uyy[b_id]]
            # ~ print ("check before", cropedDICOM.size())
            if lx[b_id] < 0:
                _, _, curentX, curentY = cropedDICOM.size()
                temTensor = torch.zeros(1, c, 0 - lx[b_id], curentY)
                if useGPU >= 0: temTensor = temTensor.cuda(useGPU)
                cropedDICOM = torch.cat((temTensor, cropedDICOM), 2)
            if ux[b_id] > h:
                _, _, curentX, curentY = cropedDICOM.size()
                temTensor = torch.zeros(1, c, ux[b_id] - h, curentY)
                if useGPU >= 0: temTensor = temTensor.cuda(useGPU)
                cropedDICOM = torch.cat((cropedDICOM, temTensor), 2)
            if ly[b_id] < 0:
                _, _, curentX, curentY = cropedDICOM.size()
                temTensor = torch.zeros(1, c, curentX, 0 - ly[b_id])
                if useGPU >= 0: temTensor = temTensor.cuda(useGPU)
                cropedDICOM = torch.cat((temTensor, cropedDICOM), 3)
            if uy[b_id] > w:
                _, _, curentX, curentY = cropedDICOM.size()
                temTensor = torch.zeros(1, c, curentX, uy[b_id] - w)
                if useGPU >= 0: temTensor = temTensor.cuda(useGPU)
                cropedDICOM = torch.cat((cropedDICOM, temTensor), 3)

            cropedDICOMs.append(cropedDICOM)
    b_crops = []
    for i in range(b):
        croped_i = torch.stack(cropedDICOMs[i::b], 1)
        b_crops.append(croped_i)
    b_crops = torch.cat(b_crops, 0)

    return b_crops


def get_hg_node_features(d, landmarks, batch_size):
    if landmarks.shape[0] != batch_size:
        landmarks = landmarks.repeat(batch_size, 1, 1)  # B,num_lands,2

    shifts = torch.tensor([[-patch_size, 0], [-patch_size, patch_size], [0, patch_size], [patch_size, patch_size],
                           [patch_size, 0], [patch_size, -patch_size], [0, patch_size], [-patch_size, -patch_size],
                           [0,  0]]).view(9, 2).to(landmarks.device)
    shifts = torch.true_divide(shifts, 255)
    # get hyer_lands
    visual_features = []
    landmarks_hyper = torch.repeat_interleave(landmarks, 9, 1)

    for i in range(landmarks.shape[1]):
        for j in range(9):
            landmarks_hyper[:, i * j, :] = landmarks_hyper[:, i * j, :] + shifts[j]
            visual_features.append(getcropedInputs(landmarks_hyper[:, i * j, :].unsqueeze(1), d,  cropSize=patch_size))

    visual_feature = torch.cat(visual_features, 1)
    init_landmark = landmarks_hyper[:, None, :, :] - landmarks_hyper[:, :, None, :]
    shape_feature = init_landmark.reshape(batch_size, landmarks_hyper.shape[1], -1)

    return visual_feature, shape_feature, landmarks_hyper



class ABHG(nn.Module):
    def __init__(self, in_ch, out_ch, num_patch=num_patch):
        super(ABHG, self).__init__()
        self.num_hyper = num_hyper
        self.H = torch.nn.Parameter((torch.ones((num_patch, self.num_hyper), requires_grad=True) / num_patch),
                                    requires_grad=True)

        self.T = torch.nn.Parameter((torch.ones((num_pt, num_patch), requires_grad=True) / num_patch),
                                    requires_grad=True)
        self.W = torch.nn.Parameter(
            (torch.ones((self.num_hyper), requires_grad=True).view(1, self.num_hyper)) / self.num_hyper,
            requires_grad=True)
        self.linear1 = nn.Linear(in_ch, 2)
        self.relu1 = nn.LeakyReLU(inplace=True)
        self.linear2 = nn.Linear(in_ch, 9)

        self.H = nn.init.normal_(self.H)
        self.T = nn.init.normal_(self.T)

        self.shifts = torch.tensor([[-patch_size, 0], [-patch_size, patch_size], [0, patch_size], [patch_size, patch_size],
                               [patch_size, 0], [patch_size, -patch_size], [0, patch_size], [-patch_size, -patch_size],
                               [0, 0]]).view(9, 2)
        self.shifts = torch.true_divide(self.shifts, 255)

    def forward(self, node_feat):
        self.shifts = self.shifts.to(node_feat.device)

        nd = node_feat
        M1 = self.T @ self.H @ torch.diag(self.W[0]) @ torch.t(self.H)

        message = torch.matmul(M1, nd)
        x1 = F.dropout(message, p=0.1, training=True)
        x1 = self.linear1(x1)
        offset = F.sigmoid(x1)  # B, 4, 2

        x2 = F.dropout(message, p=0.1, training=True)
        x2 = self.linear2(x2)  # B, 4, 9
        direction = F.softmax(x2, 2)

        final_off = offset*(direction@self.shifts)


        return final_off

class Coord_fine(nn.Module):
    def __init__(self, steps=1 , h_dim=972 + 2):
        super(Coord_fine, self).__init__()
        self.steps = steps

        self.hg = ABHG(h_dim, 2)

        self.patch_conv = nn.Conv2d(33, 33, patch_size)
        self.h_dim = 33  # +3
        self.coors = []

    def forward(self, d, x):
        bs = x.shape[0]
        updated_landmarks = x
        coors = []
        for step in range(self.steps):
            patch_feature, shape_feature, landmarks = get_hg_node_features(d, updated_landmarks, bs)
            b, num_land = patch_feature.shape[0], patch_feature.shape[1]
            patch_feature = patch_feature.view(-1, 32+1, patch_size, patch_size)
            patch_feature = self.patch_conv(patch_feature).view(b, num_land, self.h_dim)
            gin_feature = torch.cat([patch_feature, shape_feature, landmarks], -1)
            shift = self.hg(gin_feature)
            updated_landmarks = updated_landmarks + shift
            coors.append(updated_landmarks * 255)
        self.coors = coors

        return updated_landmarks * 255, self.coors



class UNET_ABHG(nn.Module):
    def __init__(self, config):
        super().__init__(config)


        self.s1 = Unet(config)
        self.s2 = Coord_fine(h_dim=33 + num_pt * 9 * 2 + 2, steps=1)
        self.pt_num = num_pt

    def get_coordinates_from_coarse_heatmaps(self, predicted_heatmap):
        global_coordinate = torch.ones(256, 256, 2).float()
        for i in range(256):
            global_coordinate[i, :, 0] = global_coordinate[i, :, 0] * i
        for i in range(256):
            global_coordinate[:, i, 1] = global_coordinate[:, i, 1] * i
        global_coordinate = (global_coordinate * torch.tensor([1 / (256 - 1), 1 / (256 - 1)])).to(
            predicted_heatmap.device)

        num_pt = predicted_heatmap.shape[1]
        bs = predicted_heatmap.shape[0]
        global_coordinate_permute = global_coordinate.permute(2, 0, 1).unsqueeze(0)
        predict = [
            torch.sum((global_coordinate_permute * predicted_heatmap[:, i:i + 1]).view(bs, 2, -1), dim=-1).unsqueeze(1)
            for i in
            range(num_pt)]
        predict = torch.cat(predict, dim=1)
        return predict

    def forward_train(self, frame):

        # coarse location stage
        s1_results = self.s1(frame)
        frame_feats, global_heatmap = s1_results['feat'], s1_results['mean']
        global_coordinate = self.get_coordinates_from_coarse_heatmaps(global_heatmap)


        # introducing noise to ensure that ABHG is better trained
        offset = torch.from_numpy(np.random.normal(loc= 0 , scale= patch_size / 256/3, size=global_coordinate.size())).float().cuda()
        global_coordinate_ = global_coordinate + offset

        frame_feats_kf = torch.cat([frame, frame_feats], 1)


        outputs, coords_kf = self.s2(frame_feats_kf, global_coordinate_)


        return global_coordinate, outputs, global_heatmap, coords_kf, s1_results['bottom']



    def forward_test(self, frame):

        # coarse location stage
        s1_results = self.s1(frame)
        frame_feats, global_heatmap = s1_results['feat'], s1_results['mean']
        frame_feats = torch.cat([frame, frame_feats], 1)
        global_coordinate = self.get_coordinates_from_coarse_heatmaps(global_heatmap)


        # ABHG fine-tuning
        local_results_t = []
        # here are some differences from the training phase
        for i in range(2):  # mc drop time
            outputs_series, coords = self.s2(frame_feats.view(-1, 33, 256, 256), global_heatmap,
                                                    global_coordinate)
            local_results_t.append(outputs_series.view(-1, 1, num_pt, 2))

        var = torch.var(torch.stack(local_results_t), 0).sum((2, 3))
        coords_series = torch.stack(local_results_t).mean(0)
        coords_series = coords_series.view(-1, 8, 1, 1)



        return global_coordinate*255, coords_series.view(1, -1, 4, 2), var


if __name__ == '__main__':
    from config import get_config

    model = UNET_ABHG(get_config()).cuda()
    frame = torch.ones((1, 1, 256, 256), dtype=torch.float).cuda() 

    model(frame)
