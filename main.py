import time
import json
import models.resnet_model as resnet_model
from cleverhans.attacks_tf import fgm
from models.madry_mnist import MadryModel
from models.aditi_mnist import AditiMNIST
from models.zico_mnist import ZicoMNIST
from utils import *
from adv_utils import *
from models.vgg16 import vgg_16
from models.acwgan_gp import ACWGAN_GP
import argparse
from matplotlib.pyplot import imsave

parser = argparse.ArgumentParser("Generative Adversarial Examples")
parser.add_argument('--dataset', type=str, default='mnist', help="数据集")
parser.add_argument('--adv', action='store_true', help="反向训练网络")
parser.add_argument('--classifier', type=str, default='resnet', help='目标神经网络')
parser.add_argument('--datapath', type=str, default='assets/data', help="数据路径")
parser.add_argument('--seed', type=int, default=1234, help="随机种子")
parser.add_argument('--batch_size', type=int, default=64, help="处理批量")
parser.add_argument('--mode', type=str, default='targeted_attack', help='模式')
parser.add_argument('--top5', action='store_true', help="top5错误")

parser.add_argument('--lr', type=float, default=1, help="学习率")
parser.add_argument('--n_adv_examples', type=int, default=1000000,
                    help="对抗样本批次数量")
parser.add_argument('--n_iters', type=int, default=1000,
                    help="计算对抗样本的内部迭代次数")
parser.add_argument('--z_dim', type=int, default=128, help="噪音向量维数")
parser.add_argument('--checkpoint_dir', type=str, default='assets/checkpoint',
                    help='存checkpoint')
parser.add_argument('--result_dir', type=str, default='assets/results',
                    help='存生成图像')
parser.add_argument('--log_dir', type=str, default='assets/logs',
                    help='存log')
parser.add_argument('--source', type=int, default=0, help="真实分类")
parser.add_argument('--target', type=int, default=1, help="目标分类")
parser.add_argument('--lambda1', type=float, default=100, help="紧密正则项的系数")
parser.add_argument('--lambda2', type=float, default=100, help="斥力正则项的系数")
parser.add_argument('--n2collect', type=int, default=1024, help="要收集的对抗样本数量")
parser.add_argument('--eps', type=float, default=0.1, help="噪声增强的攻击Eps")
parser.add_argument('--noise', action="store_true", help="为攻击添加噪声增强")
parser.add_argument('--z_eps', type=float, default=0.1, help="潜空间搜索区域的软约束")
parser.add_argument('--adv_gen', action="store_true", help="使用生成对抗样本进行对抗训练")
parser.add_argument('--trained', action="store_true", help="培训模型")

args = parser.parse_args()


def resnet_template(images, training, hps):
    # Do per image standardization
    images_standardized = per_image_standardization(images)
    model = resnet_model.ResNet(hps, images_standardized, training)
    model.build_graph()
    return model.logits


def vgg_template(images, training, hps):
    images_standardized = per_image_standardization(images)
    logits, _ = vgg_16(images_standardized, num_classes=hps.num_classes, is_training=training, dataset=hps.dataset)
    return logits


def madry_template(images, training):
    model = MadryModel(images)
    return model.pre_softmax


def aditi_template(images, training):
    model = AditiMNIST(images)
    return model.logits


def zico_template(images, training):
    model = ZicoMNIST(images)
    return model.logits


def targeted_attack(hps, source, target, lambda1, lambda2, noise=False):

    source_np = np.asarray([source] * args.batch_size).astype(np.int32)
    target_np = np.asarray([target] * args.batch_size).astype(np.int32)

    if args.classifier == "madry":
        net = tf.make_template('net', madry_template)
    elif args.classifier == 'aditi':
        net = tf.make_template('net', aditi_template)
    elif args.classifier == 'zico':
        net = tf.make_template('net', zico_template)
    else:
        net = tf.make_template('net', resnet_template, hps=hps) if args.classifier == 'resnet' else \
            tf.make_template('net', vgg_template, hps=hps)

    adv_noise = tf.get_variable('adv_noise', shape=(args.batch_size, args.image_size, args.image_size, args.channels),
                                dtype=tf.float32, initializer=tf.zeros_initializer)
    adv_z = tf.get_variable('adv_z',
                            shape=(args.batch_size, args.z_dim),
                            dtype=tf.float32,
                            initializer=tf.random_normal_initializer)

    ref_z = tf.get_variable('ref_z',
                            shape=(args.batch_size, args.z_dim),
                            dtype=tf.float32,
                            initializer=tf.zeros_initializer)

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    dim_D = 32
    dim_G = 32

    acgan = ACWGAN_GP(
        sess,
        epoch=10,
        batch_size=args.batch_size,
        z_dim=args.z_dim,
        dataset_name=args.dataset,
        checkpoint_dir=args.checkpoint_dir,
        result_dir=args.result_dir,
        log_dir=args.log_dir,
        dim_D=dim_D,
        dim_G=dim_G
    )

    acgan.build_model()

    adv_images = acgan.generator(adv_z, source_np, reuse=True)
    _, acgan_logits = acgan.discriminator(adv_images, update_collection=None, reuse=True)
    acgan_pred = tf.argmax(acgan_logits, axis=1)
    acgan_softmax = tf.nn.softmax(acgan_logits)

    if noise:
        adv_images += args.eps * tf.tanh(adv_noise)
        adv_images = tf.clip_by_value(adv_images, clip_value_min=0., clip_value_max=1.0)

    net_logits = net(adv_images, training=False)
    net_softmax = tf.nn.softmax(net_logits)
    net_pred = tf.argmax(net_logits, axis=1)

    obj = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=net_logits, labels=target_np)) + \
          lambda1 * tf.reduce_mean(tf.maximum(tf.square(ref_z - adv_z) - args.z_eps ** 2, 0.0)) + \
          lambda2 * tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(logits=acgan_logits, labels=source_np))

    _iter = tf.placeholder(tf.float32, shape=(), name="iter")
    with tf.variable_scope("train_ops"):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=args.lr)
        var = 0.01 / (1. + _iter) ** 0.55
        if noise:
            grads = optimizer.compute_gradients(obj, var_list=[adv_z, adv_noise])
        else:
            grads = optimizer.compute_gradients(obj, var_list=[adv_z])

        new_grads = []
        for grad, v in grads:
            if v is not adv_noise:
                new_grads.append((grad + tf.random_normal(shape=grad.get_shape().as_list(), stddev=tf.sqrt(var)), v))
            else:
                new_grads.append((grad / tf.sqrt(tf.reduce_sum(tf.square(grad), axis=[1, 2, 3], keep_dims=True)), v))

        adv_op = optimizer.apply_gradients(new_grads)

    momentum_init = tf.variables_initializer(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='train_ops'))
    init_op = tf.group(momentum_init, tf.variables_initializer([adv_z, adv_noise]))
    with tf.control_dependencies([init_op]):
        init_op = tf.group(init_op, tf.assign(ref_z, adv_z))

    sess.run(tf.global_variables_initializer())

    save_path, save_path_ckpt = get_weights_path(args)
    print(save_path,save_path_ckpt)
    if args.classifier == 'madry':
        if args.trained:
            saver4classifier = tf.train.Saver(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="net"))
        else:
            saver4classifier = tf.train.Saver(
                {x.name[4:-2]: x for x in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="net")})
    else:
        saver4classifier = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='net'))
    checkpoint_dir = os.path.join(args.checkpoint_dir, acgan.model_dir, acgan.model_name)
    try:
        ckpt_state = tf.train.get_checkpoint_state(save_path)
    except tf.errors.OutOfRangeError as e:
        print('[!] Cannot restore checkpoint: %s' % e)
    if not (ckpt_state and ckpt_state.model_checkpoint_path):
        print('[!] No model to eval yet at %s' % save_path)
    print('[*] Loading checkpoint %s' % ckpt_state.model_checkpoint_path)
    saver4classifier.restore(sess, ckpt_state.model_checkpoint_path)

    saver4gen = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                                          scope='generator|discriminator|classifier'))
    try:
        ckpt_state = tf.train.get_checkpoint_state(checkpoint_dir)
    except tf.errors.OutOfRangeError as e:
        print('[!] Cannot restore checkpoint: %s' % e)
    if not (ckpt_state and ckpt_state.model_checkpoint_path):
        print('[!] No model to eval yet at %s' % checkpoint_dir)
    print('[*] Loading checkpoint %s' % ckpt_state.model_checkpoint_path)
    saver4gen.restore(sess, ckpt_state.model_checkpoint_path)

    acc = 0.
    adv_acc = 0.
    adv_im_np = []
    latent_z = []
    init_latent_z = []
    for batch in range(args.n_adv_examples):
        sess.run(init_op)
        preds_np, probs_np, im_np, cost_before = sess.run([net_pred, net_softmax, adv_images, obj])


        for i in range(args.n_iters):
            _, now_cost, pred_np, acgan_pred_np, acgan_probs = sess.run(
                [adv_op, obj, net_pred, acgan_pred, acgan_softmax],
                feed_dict={_iter: i})
            ok = np.logical_and(pred_np == target, acgan_pred_np == source)
            print("   [*] {}th iter, cost: {}, success: {}/{}".format(i + 1, now_cost, np.sum(ok), args.batch_size))

        adv_preds_np, acgan_preds_np, adv_probs_np, acgan_probs_np, im_np, hidden_z, init_z, cost_after = sess.run(
            [net_pred, acgan_pred,
             net_softmax, acgan_softmax, adv_images, adv_z, ref_z, obj])
        acc += np.sum(preds_np == source)
        idx = np.logical_and(adv_preds_np == target, acgan_preds_np == source)
        adv_acc += np.sum(idx)
        adv_im_np.extend(im_np[idx])
        latent_z.extend(hidden_z[idx])
        init_latent_z.extend(init_z[idx])
        print("batch: {}, acc: {}, adv_acc: {}, num collected: {}, cost before: {}, cost after: {}".
              format(batch + 1, acc / ((batch + 1) * args.batch_size), adv_acc / ((batch + 1) * args.batch_size),
                     len(adv_im_np), cost_before, cost_after))

        if len(adv_im_np) >= args.n2collect:
            adv_im_np = np.asarray(adv_im_np)
            latent_z = np.asarray(latent_z)
            size = int(np.sqrt(args.n2collect))
            classifier = args.classifier
            if args.adv:
                classifier += '_adv'

            folder_format = '{}_{}_targeted_attack_with_z0'
            if noise: folder_format += '_noise'
            np.savez(os.path.join(check_folder(folder_format.format(args.dataset, classifier)),
                                  'from{}to{}'.format(source, target)), adv_imgs=adv_im_np, latent_z=latent_z,
                     init_latent_z=init_latent_z)
            save_images(adv_im_np[:args.n2collect, :, :, :], [size, size],
                        os.path.join(check_folder(folder_format.format(args.dataset, classifier)),
                                     '{}_ims_from{}_to{}.png').format(args.dataset, source, target))
            break


def main():
    np.random.seed(args.seed)
    tf.compat.v1.set_random_seed(args.seed)

    num_classes = 10
    args.num_classes = 10
    args.image_size = 28
    args.channels = 1

    print("[*] input args:\n", json.dumps(vars(args), indent=4, separators=(',', ':')))

    num_residual_units = 5
    hps = resnet_model.HParams(batch_size=args.batch_size,
                               num_classes=num_classes,
                               min_lrn_rate=0.0001,
                               lrn_rate=0.1,
                               num_residual_units=num_residual_units,
                               use_bottleneck=False,
                               weight_decay_rate=0.0002,
                               relu_leakiness=0.1,
                               optimizer='mom',
                               dataset=args.dataset)


    targeted_attack(hps, args.source, args.target, args.lambda1, args.lambda2, noise=args.noise)


if __name__ == '__main__':
    main()
