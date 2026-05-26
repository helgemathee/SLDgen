from SLDgen import config
from SLDgen.run import run, save_config, set_error_logging

if __name__ == "__main__":
    args = config.parse_arguments()

    if not args.debug:
        set_error_logging()

    run(args)

    save_config(args)
