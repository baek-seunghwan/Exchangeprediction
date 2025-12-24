import torch
from torch.nn.utils import clip_grad_norm_


class Optim:
    """
    Wrapper optimizer used in train_test.py

    사용법:
        optim = Optim(
            model.parameters(), args.optim, lr, args.clip, 
            weight_decay=args.weight_decay, 
            lr_gamma=None
        )

        ...
        grad_norm = optim.step()
    """

    def __init__(self, params, optim: str, lr: float, clip: float, 
                 weight_decay: float = 0.0, 
                 lr_gamma: float = None):
        """
        params      : model.parameters()
        optim       : 'adam', 'sgd', 'rmsprop' 등
        lr          : learning rate
        clip        : gradient clipping max norm
        weight_decay: L2 regularization (optimizer에만 전달)
        lr_gamma    : 학습률 감소 비율 (None 이면 사용 안 함, 0<gamma<1 범위)
        """
        self.lr = lr
        self.clip = clip

        optim = optim.lower()
        if optim == "sgd":
            self.optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
        elif optim == "adam":
            self.optimizer = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
        elif optim == "rmsprop":
            self.optimizer = torch.optim.RMSprop(params, lr=lr, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unsupported optimizer type: {optim}")

        # clipping 대상 파라미터들 저장
        self.params = list(self.optimizer.param_groups[0]["params"])

        # lr_gamma가 0<gamma<1 범위일 때만 지수 감소 스케줄러 사용
        if lr_gamma is not None and 0.0 < lr_gamma < 1.0:
            self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
                self.optimizer, gamma=lr_gamma
            )
        else:
            self.scheduler = None

    def step(self):
        """
        gradient clipping + optimizer.step()

        return: grad_norm (없으면 0.0)
        """
        grad_norm = 0.0

        # gradient clipping
        if self.clip is not None and self.clip > 0:
            grad_norm = float(clip_grad_norm_(self.params, self.clip))

        # 파라미터 업데이트
        self.optimizer.step()

        return grad_norm

    def lr_step(self):
        """
        Learning rate scheduler step (epoch 단위에서만 호출)
        """
        if self.scheduler is not None:
            self.scheduler.step()

    def get_lr(self) -> float:
        """현재 학습률 반환"""
        return float(self.optimizer.param_groups[0]["lr"])