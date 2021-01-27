import argparse
import os

from training.train_loop import validation

#os.environ["CUDA_VISIBLE_DEVICES"] = '1'

# ----------------------------------------------------------------------------
def _str_to_bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

# ----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description='DCGAN',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--load_path', help='model load path', required=True)
    parser.add_argument('--generate_num', help='The number of generated images', required=True)
    parser.add_argument('--generate_num', help='The number of generated images', default=16)
    parser.add_argument('--seed', help='Set seed', default=22222)

    
    args = parser.parse_args()

    validation(**vars(args))


# ----------------------------------------------------------------------------
if __name__ == "__main__":
    main()
    

# ----------------------------------------------------------------------------

