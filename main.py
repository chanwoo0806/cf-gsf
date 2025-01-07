from cf_gsf.configurator import args
from cf_gsf.dataloader import DataHandler
from cf_gsf.util import init_seed, Logger, build_model, build_evaluator

def main():
    init_seed()
    logger = Logger()
    data_handler = DataHandler()
    data_handler.load_data()
    model = build_model(data_handler).to(args.device)
    evaluator = build_evaluator(data_handler, logger)
    evaluator.test(model)

if __name__ == '__main__':
    main()