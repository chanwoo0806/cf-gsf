from cf_gsf.metric import Metric
from cf_gsf.configurator import args
from cf_gsf.util import log_exceptions

class BaseEvaluator:
    def __init__(self, data_handler, logger):
        self.data_handler = data_handler
        self.logger = logger
        self.metric = Metric()

    @log_exceptions
    def test(self, model):
        model.eval()
        eval_result = self.metric.eval(model, self.data_handler.test_dataloader)
        self.logger.log("[TEST]")
        self.logger.log_eval(eval_result, args.ks)
        if args.summary is not None:
            self.logger.log_summary(eval_result, args.ks)