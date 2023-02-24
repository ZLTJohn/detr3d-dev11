# path:/home/zhenglt/mmdev11/mmengine/optim/optimizer/optimizer_wrapper.py
def update_params(self,
                      loss: torch.Tensor,
                      step_kwargs: Optional[Dict] = None,
                      zero_kwargs: Optional[Dict] = None) -> None:
        """Update parameters in :attr:`optimizer`.

        Args:
            loss (torch.Tensor): A tensor for back propagation.
            step_kwargs (dict): Arguments for optimizer.step.
                Defaults to None.
                New in version v0.4.0.
            zero_kwargs (dict): Arguments for optimizer.zero_grad.
                Defaults to None.
                New in version v0.4.0.
        """
        # from time import time
        # torch.cuda.synchronize()
        # _=time()
        if step_kwargs is None:
            step_kwargs = {}
        if zero_kwargs is None:
            zero_kwargs = {}
        loss = self.scale_loss(loss)
        # torch.cuda.synchronize()
        # __ = time()
        # print('\tscale loss', __-_)
        self.backward(loss)
        torch.cuda.synchronize()
        ___=time()
        print('\tbackward', ___-__)
        # Update parameters only if `self._inner_count` is divisible by
        # `self._accumulative_counts` or `self._inner_count` equals to
        # `self._max_counts`
        if self.should_update():
            self.step(**step_kwargs)
            torch.cuda.synchronize()
            ____=time()
            print('\tstep',____-___)
            self.zero_grad(**zero_kwargs)
            torch.cuda.synchronize()
            _____=time()
            print('\tzero grad',_____-____)

# path: /home/zhenglt/mmdev11/mmengine/model/wrappers/distributed.py
def train_step(self, data: Union[dict, tuple, list],
                   optim_wrapper: OptimWrapper) -> Dict[str, torch.Tensor]:
        """Interface for model forward, backward and parameters updating during
        training process.

        :meth:`train_step` will perform the following steps in order:

        - If :attr:`module` defines the preprocess method,
          call ``module.preprocess`` to pre-processing data.
        - Call ``module.forward(**data)`` and get losses.
        - Parse losses.
        - Call ``optim_wrapper.optimizer_step`` to update parameters.
        - Return log messages of losses.

        Args:
            data (dict or tuple or list): Data sampled from dataset.
            optim_wrapper (OptimWrapper): A wrapper of optimizer to
                update parameters.

        Returns:
            Dict[str, torch.Tensor]: A ``dict`` of tensor for logging.
        """
        # Enable automatic mixed precision training context.
        from time import time
        torch.cuda.synchronize()
        _=time()
        with optim_wrapper.optim_context(self):
            data = self.module.data_preprocessor(data, training=True)
            losses = self._run_forward(data, mode='loss')
        torch.cuda.synchronize()
        __ = time()
        print('forward time', __-_)
        parsed_loss, log_vars = self.module.parse_losses(losses)
        optim_wrapper.update_params(parsed_loss)
        torch.cuda.synchronize()
        print('update param time', time()-__)
        if self.detect_anomalous_params:
            detect_anomalous_params(parsed_loss, model=self)
        return log_vars