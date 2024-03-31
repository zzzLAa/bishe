import argparse
from utils import *
from models.acwgan_gp import ACWGAN_GP
import json

parser = argparse.ArgumentParser("Training Wasserstein ACGAN")
parser.add_argument('--dataset', type=str, default='mnist', help="数据集 ")
parser.add_argument('--n_epochs', type=int, default=50, help="训练轮数")
parser.add_argument('--batch_size', type=int, default=64, help="处理批次")
parser.add_argument('--z_dim', type=int, default=128, help="噪声向量的维数")
parser.add_argument('--checkpoint_dir', type=str, default='assets/checkpoint',
                    help='存储checkpoint')
parser.add_argument('--result_dir', type=str, default='assets/results',
                    help='存储生成图像')
parser.add_argument('--log_dir', type=str, default='assets/logs',
                    help='存储logs')

args = parser.parse_args()
check_folder(args.checkpoint_dir)

# --result_dir
check_folder(args.result_dir)

# --result_dir
check_folder(args.log_dir)

def main():
    print("[*] input args:\n", json.dumps(vars(args), indent=4, separators=(',', ':')))

    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    dim_D = 32
    dim_G = 32


    with tf.Session(config=tf_config) as sess:
        gan = ACWGAN_GP(
            sess,
            epoch=args.n_epochs,
            batch_size=args.batch_size,
            z_dim=args.z_dim,
            dataset_name=args.dataset,
            checkpoint_dir=args.checkpoint_dir,
            result_dir=args.result_dir,
            log_dir=args.log_dir,
            dim_D=dim_D,
            dim_G=dim_G
        )

        gan.build_model()
        show_all_variables()
        gan.train()
        print(" [*] Training finished")
        gan.visualize_results(args.n_epochs - 1)
        print(" [*] Testing finished")


if __name__ == '__main__':
    main()
