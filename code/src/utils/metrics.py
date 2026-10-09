import torch
import torch.nn.functional as F


def update_iou(pred, target, num_classes, intersection, union, ignore_index=255):
    """
    Accumulate class-wise intersection and union for semantic segmentation.
    Pixels with the ignore index are excluded from metric computation.
    """

    valid = target != ignore_index

    for cls in range(num_classes):
        pred_cls = (pred == cls) & valid
        target_cls = (target == cls) & valid

        intersection[cls] += (pred_cls & target_cls).sum()

        union[cls] += (pred_cls | target_cls).sum()

    return intersection, union


def compute_miou(intersection, union):
    """
    Compute mIoU across classes.
    """

    valid_classes = union > 0
    iou = intersection[valid_classes] / union[valid_classes]

    return iou.mean()


def update_depth_metrics(pred, target, sq_error_sum, abs_rel_sum, valid_count):
    """
    Accumulate squared error and absolute relative error over valid depth pixels.
    Only target depth values greater than zero are included.
    """

    valid = target > 0

    pred = pred[valid]
    target = target[valid]

    sq_error_sum += torch.sum((pred - target) ** 2).item()
    abs_rel_sum += torch.sum(torch.abs(pred - target) / target).item()
    valid_count += target.numel()

    return sq_error_sum, abs_rel_sum, valid_count


def compute_rmse(sq_error_sum, valid_count):
    """
    Compute root mean square error.
    """

    return (sq_error_sum / valid_count) ** 0.5


def compute_abs_rel(abs_rel_sum, valid_count):
    """
    Compute absolute relative error.
    """

    return abs_rel_sum / valid_count


def compute_gram_mse(clean_patch_tokens, adv_patch_tokens):
    """

    """

    clean_gram = compute_gram(clean_patch_tokens)
    adv_gram = compute_gram(adv_patch_tokens)

    gram_mse = F.mse_loss(adv_gram, clean_gram).item()

    return gram_mse


def compute_gram(patch_tokens):
    """

    """

    # [B, P, D]
    x = F.normalize(patch_tokens.float(), dim=-1)

    # [B, P, P]
    gram = torch.matmul(x, x.transpose(-1, -2))

    return gram


def compute_off_diagonal(gram):
    """

    """

    P = gram.shape[-1]

    off_diag_mask = ~torch.eye(
        P,
        dtype=torch.bool,
        device=gram.device
    )

    return gram[:, off_diag_mask]


def compute_normalised_gram_difference(clean_patch_tokens, adv_patch_tokens):
    """

    """

    clean_gram = compute_gram(clean_patch_tokens)
    adv_gram = compute_gram(adv_patch_tokens)

    clean_offdiag = compute_off_diagonal(clean_gram)
    adv_offdiag = compute_off_diagonal(adv_gram)

    difference = torch.linalg.vector_norm(clean_offdiag - adv_offdiag, dim=-1)

    clean_magnitude = torch.linalg.norm(clean_offdiag, dim=-1)

    normalised_difference = difference / (clean_magnitude + 1e-8)

    return normalised_difference.mean().item()
