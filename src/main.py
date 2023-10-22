import os, argparse
import os.path as osp
import sonata
from utils.utils import *
from utils.vis import *
import scotv1
import yaml
from easydict import EasyDict

def main(cfg):
    data1 = load_data(cfg.data_path.dataset_url1)
    data2 = load_data(cfg.data_path.dataset_url2)
    label1 = load_data(cfg.data_path.label_url1)
    label2 = load_data(cfg.data_path.label_url2)
    basename = osp.basename(osp.dirname(cfg.data_path.dataset_url1))

    # visualize modalities
    if basename in ['swiss_roll']:
        plt_domain(data1, color='#009ACD', title='domain1', save_url=osp.join(cfg.save_path, 'domain1.png'))
        plt_domain(data2, color='#FF8C00', title='domain2', save_url=osp.join(cfg.save_path, 'domain2.png'))
    elif basename in ['circle']:
        plt_domain_by_const_label(data1, label1,  color='Blues', title='domain1', save_url=osp.join(cfg.save_path, 'domain1.png'))
        plt_domain_by_const_label(data2, label2,  color='Oranges', title='domain2', save_url=osp.join(cfg.save_path, 'domain2.png'))
    else:
        plt_domain_by_label(data1, label1, color='#009ACD', title='domain1', save_url=osp.join(cfg.save_path, 'domain1.png'))
        plt_domain_by_label(data2, label2, color='#FF8C00', title='domain2', save_url=osp.join(cfg.save_path, 'domain2.png'))


    # mapping from SCOT, also could be any other manifold aligners
    scot = scotv1.SCOT(data1.copy(), data2.copy())
    scot.align(k = cfg.scot.k, e=cfg.scot.e, mode=cfg.scot.mode, metric=cfg.scot.metric, normalize=cfg.scot.normalize)
    mapping = scot.coupling
    x_aligned, y_aligned = projection_barycentric(scot.X, scot.y, mapping, XontoY = cfg.plt.XontoY)

    # visualize mapping
    if basename in ['swiss_roll']:  # for non-ambiguity
        plt_mapping(x_aligned, y_aligned, save_url=osp.join(cfg.save_path, 'mapping.png'))
    elif basename in ['circle']:  # for all-ambiguity
        plt_mapping_by_const_label(x_aligned, y_aligned, label1, label2, save_url=osp.join(cfg.save_path, 'mapping.png'))
    else: # for partial-ambiguity
        plt_mapping_by_label(x_aligned, y_aligned, label1, label2, save_url=osp.join(cfg.save_path, 'mapping.png'))



    # cell-cell alternaltive mappings from SONATA
    sn = sonata.sonata(cfg.sonata)
    alter_mappings = sn.alt_mapping(data=data1) 

    if alter_mappings != None:
        for idx, m in enumerate(alter_mappings, start=1):
            this_mapping = np.matmul(m, mapping)
            x_aligned, y_aligned = projection_barycentric(scot.X, scot.y, this_mapping, XontoY = cfg.plt.XontoY)

            # visualize alternaltive mapping
            if basename in ['swiss_roll']:  # for non-ambiguity
                plt_mapping(x_aligned, y_aligned, save_url=osp.join(cfg.save_path, 'alter_mapping{}.png'.format(idx)))
            elif basename in ['circle']:  # for all-ambiguity
                plt_mapping_by_const_label(x_aligned, y_aligned, label1, label2, save_url=osp.join(cfg.save_path, 'alter_mapping{}.png'.format(idx)))
            else: # for partial-ambiguity
                plt_mapping_by_label(x_aligned, y_aligned, label1, label2, save_url=osp.join(cfg.save_path, 'alter_mapping{}.png'.format(idx)))
        



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, help='The url of config file')
    args = parser.parse_args()

    with open(args.cfg, 'r',encoding='utf8') as file:
        cfg = EasyDict(yaml.safe_load(file))

    main(cfg)