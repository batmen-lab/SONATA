import os
import yaml
from sklearn.preprocessing import normalize
import argparse

import sys
sys.path.insert(1, '../src/')
import sonata
from run_baselines.scot import scotv1
from utils.utils import *
from utils.vis import *


def get_preprocessed_data(params):
    data_url1 = os.path.join(params["data_path"], params["data_url1"])
    data_url2 = os.path.join(params["data_path"], params["data_url2"])
    assert os.path.isfile(data_url1) and os.path.isfile(data_url2)

    label_url1 = os.path.join(params["data_path"], params["label_url1"])
    label_url2 = os.path.join(params["data_path"], params["label_url2"])
    assert os.path.isfile(label_url1) and os.path.isfile(label_url2)

    data1 = load_data(data_url1, )
    data2 = load_data(data_url2, )
    print("data size: data1 = {}, data2 = {}".format(data1.shape, data2.shape))

    label1 = load_data(label_url1, ).astype(int)
    label2 = load_data(label_url2, ).astype(int)

    if params["norm"]: 
            data1 = normalize(data1, norm=params["norm"])
            data2 = normalize(data2, norm=params["norm"])

    if params["sort_label"]: 
            data1, label1, _ = sorted_by_label(data1, label1)
            data2, label2, _ = sorted_by_label(data2, label2)

    print("data1 shape={}\tdata2 shape={}".format(data1.shape, data2.shape))
    print("label1 shape={}\tlabel2 shape={}".format(label1.shape, label2.shape))
    return data1, data2, label1, label2


def main(args):
    fname = f"../examples/cfgs/{args.dataset}.yaml"
    save_path = f"../results/scalability/{args.dataset}"
    
    # load parameters and datasets
    with open(fname, "r") as file:
            params = yaml.safe_load(file)
    data1, data2, label1, label2 = get_preprocessed_data(params)
    
    sn1 = sonata.sonata(params)
    DiagnoseResult1 = sn1.diagnose(data1, save_dir=os.path.join(params['save_dir'], "Modality1"))
    plt_cannotlink_by_labelcolor(data1, DiagnoseResult1.ambiguous_idx, DiagnoseResult1.ambiguous_labels, DiagnoseResult1.cannot_links, alpha=0.6, cl_alpha = 0.1, marker='.',
                                    curve_link=True, save_url = f"{save_path}/Modality1/diagnose_result.png", show=False)

    sn2 = sonata.sonata(params)
    DiagnoseResult2 = sn2.diagnose(data2, save_dir=os.path.join(params['save_dir'], "Modality2"))
    plt_cannotlink_by_labelcolor(data2, DiagnoseResult2.ambiguous_idx, DiagnoseResult2.ambiguous_labels, DiagnoseResult2.cannot_links, alpha=0.6, cl_alpha = 0.1, marker='.',
                                    curve_link=True, save_url = f"{save_path}/Modality2/diagnose_result.png", show=False)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sonata Test")
    parser.add_argument("--dataset", type=str, default="t_branch", help="Dataset name")
    args = parser.parse_args()

    main(args)