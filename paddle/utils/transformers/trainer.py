from transformers import IntervalStrategy, Trainer
from transformers.integrations import TensorBoardCallback


class MultiLossTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.
        Subclass and override for custom behavior.
        """
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        outputs = model(**inputs)
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            loss = self.label_smoother(outputs, labels)
        else:
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
            if isinstance(loss, dict):
                loss_ = 0
                log = {}
                for key, value in loss.items():
                    loss_ += value
                    log[f"loss_{key}"] = value.detach().item()
                if self.args.logging_strategy == IntervalStrategy.STEPS and \
                        self.state.global_step % self.args.logging_steps == 0:
                    self.control.should_log = True
                    self.log(log)
                else:  # nonstop logging to TensorBoard
                    for cb in self.callback_handler.callbacks:
                        if isinstance(cb, TensorBoardCallback):
                            ctrl = cb.on_log(self.args, self.state, self.control, log)
                            if ctrl is not None:
                                self.control = ctrl
                loss = loss_

        return (loss, outputs) if return_outputs else loss
