import torch
import torch.nn as nn


class KeypointDistilationLoss(nn.Module):
    """
    Keypoint Knowledge Distilation loss module.
    Args:
        loss_type (str): Choose between l1 and l2 losses.
    """

    def __init__(self, loss_type: str = "l1"):
        super(KeypointDistilationLoss, self).__init__()
        if loss_type == "l1":
            self.loss_fn = nn.L1Loss(reduction="none")
        elif loss_type == "l2":
            self.loss_fn = nn.MSELoss(reduction="none")
        else:
            raise NotImplementedError("Unsupported loss function")

    def forward(self, pred_outputs: tuple, teacher_outputs: tuple, pelvis_id: int = 0):
        """
        Compute Knowledge distilation loss, this includes loss for 3D keypoints loss, and 2D keypoints loss.
        Args:
            pred_outputs (tuple): Tuple of shape (1, 2) contains predicted 3D keypoints, 2D joint keypoints.
            teacher_outputs (tuple): Tuple of shape (1, 2) contains teacher corresponding boolean tensor for the parameters, 3D keypoints, and 2D keypoints.
        Returns:
            torch.Tensor: Distilation Loss.
        """

        pred_keypoints_2d, pred_keypoints_3d = pred_outputs
        teacher_keypoints_2d, teacher_keypoints_3d = teacher_outputs

        # keypoints 3d

        # normalizing all keypoints with the pelvis position (root joint)
        teacher_keypoints_3d = teacher_keypoints_3d - teacher_keypoints_3d[
            :, pelvis_id, :
        ].unsqueeze(dim=1)
        pred_keypoints_3d = pred_keypoints_3d - pred_keypoints_3d[
            :, pelvis_id, :
        ].unsqueeze(dim=1)
        keypoints_3d_loss = self.loss_fn(pred_keypoints_3d, teacher_keypoints_3d).sum(
            dim=(0, 1, 2)
        )

        # keypoints 2d
        keypoints_2d_loss = self.loss_fn(pred_keypoints_2d, teacher_keypoints_2d).sum(
            dim=(0, 1, 2)
        )

        return (keypoints_2d_loss, keypoints_3d_loss)


class FeatureMapDistillationLoss(nn.Module):
    def __init__(self, loss_type='mse'):
        """Backbone feature-maps loss module.

        Args:
            loss_type (str, optional): Loss function to use for calculating loss. Defaults to 'mse'.
        """
        super(FeatureMapDistillationLoss, self).__init__()
        if loss_type == 'mse':
            self.loss_fn = nn.MSELoss()
        elif loss_type == 'cosine':
            self.loss_fn = nn.CosineEmbeddingLoss()
        else:
            raise ValueError("Unsupported loss type. Choose 'mse' or 'cosine'.")

    def forward(self, teacher_features, student_features):
        """Calculate loss on the feature maps.

        Args:
            teacher_features (torch.Tensor): Teacher feature maps.
            student_features (torch.Tensor): Student feature maps.

        Returns:
            torch.Tensor: Feature map loss.
        """
        if isinstance(self.loss_fn, nn.CosineEmbeddingLoss):
            # Cosine similarity requires an additional target (1 for similarity)
            batch_size = teacher_features.size(0)
            target = torch.ones(batch_size).to(teacher_features.device)
            return self.loss_fn(teacher_features.flatten(1), student_features.flatten(1), target)
        else:
            return self.loss_fn(teacher_features, student_features)

        

class Keypoint2DLoss(nn.Module):

    def __init__(self, loss_type: str = "l1"):
        """
        2D keypoint loss module.
        Args:
            loss_type (str): Choose between l1 and l2 losses.
        """
        super(Keypoint2DLoss, self).__init__()
        if loss_type == "l1":
            self.loss_fn = nn.L1Loss(reduction="none")
        elif loss_type == "l2":
            self.loss_fn = nn.MSELoss(reduction="none")
        else:
            raise NotImplementedError("Unsupported loss function")

    def forward(
        self, pred_keypoints_2d: torch.Tensor, gt_keypoints_2d: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute 2D reprojection loss on the keypoints.
        Args:
            pred_keypoints_2d (torch.Tensor): Tensor of shape [B, S, N, 2] containing projected 2D keypoints (B: batch_size, S: num_samples, N: num_keypoints)
            gt_keypoints_2d (torch.Tensor): Tensor of shape [B, S, N, 3] containing the ground truth 2D keypoints and confidence.
        Returns:
            torch.Tensor: 2D keypoint loss.
        """
        conf = gt_keypoints_2d[:, :, -1].unsqueeze(-1).clone()
        batch_size = conf.shape[0]
        loss = (conf * self.loss_fn(pred_keypoints_2d, gt_keypoints_2d[:, :, :-1])).sum(
            dim=(1, 2)
        )
        return loss.sum()


class Keypoint3DLoss(nn.Module):

    def __init__(self, loss_type: str = "l1"):
        """
        3D keypoint loss module.
        Args:
            loss_type (str): Choose between l1 and l2 losses.
        """
        super(Keypoint3DLoss, self).__init__()
        if loss_type == "l1":
            self.loss_fn = nn.L1Loss(reduction="none")
        elif loss_type == "l2":
            self.loss_fn = nn.MSELoss(reduction="none")
        else:
            raise NotImplementedError("Unsupported loss function")

    def forward(
        self,
        pred_keypoints_3d: torch.Tensor,
        gt_keypoints_3d: torch.Tensor,
        pelvis_id: int = 0,
    ):
        """
        Compute 3D keypoint loss.
        Args:
            pred_keypoints_3d (torch.Tensor): Tensor of shape [B, S, N, 3] containing the predicted 3D keypoints (B: batch_size, S: num_samples, N: num_keypoints)
            gt_keypoints_3d (torch.Tensor): Tensor of shape [B, S, N, 4] containing the ground truth 3D keypoints and confidence.
        Returns:
            torch.Tensor: 3D keypoint loss.
        """
        batch_size = pred_keypoints_3d.shape[0]
        gt_keypoints_3d = gt_keypoints_3d.clone()
        pred_keypoints_3d = pred_keypoints_3d - pred_keypoints_3d[
            :, pelvis_id, :
        ].unsqueeze(dim=1)
        gt_keypoints_3d[:, :, :-1] = gt_keypoints_3d[:, :, :-1] - gt_keypoints_3d[
            :, pelvis_id, :-1
        ].unsqueeze(dim=1)
        conf = gt_keypoints_3d[:, :, -1].unsqueeze(-1).clone()
        gt_keypoints_3d = gt_keypoints_3d[:, :, :-1]
        loss = (conf * self.loss_fn(pred_keypoints_3d, gt_keypoints_3d)).sum(dim=(1, 2))
        return loss.sum()


class ParameterLoss(nn.Module):

    def __init__(self):
        """
        MANO parameter loss module.
        """
        super(ParameterLoss, self).__init__()
        self.loss_fn = nn.MSELoss(reduction="none")

    def forward(
        self, pred_param: torch.Tensor, gt_param: torch.Tensor, has_param: torch.Tensor
    ):
        """
        Compute MANO parameter loss.
        Args:
            pred_param (torch.Tensor): Tensor of shape [B, S, ...] containing the predicted parameters (body pose / global orientation / betas)
            gt_param (torch.Tensor): Tensor of shape [B, S, ...] containing the ground truth MANO parameters.
        Returns:
            torch.Tensor: L2 parameter loss loss.
        """
        batch_size = pred_param.shape[0]
        num_dims = len(pred_param.shape)
        mask_dimension = [batch_size] + [1] * (num_dims - 1)
        has_param = has_param.type(pred_param.type()).view(*mask_dimension)
        loss_param = has_param * self.loss_fn(pred_param, gt_param)
        return loss_param.sum()
