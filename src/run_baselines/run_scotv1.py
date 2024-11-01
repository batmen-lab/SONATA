"""
Modifications to SCOT v1 for testing:

1. Line 157: Corrected parameter passing in the construct_graph function:
   - Changed from `self.construct_graph(k, mode="connectivity", metric="correlation")`
   - To `self.construct_graph(k, mode=mode, metric=metric)`

Usage:
    cd src
    python run_baselines/run_scotv1.py --dataset t_branch
   
"""
from scot import scotv1
import numpy as np
import os, argparse, sys
from sklearn.preprocessing import normalize
import yaml

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)
from utils.utils import load_data, sorted_by_label
from utils.vis import plt_heatmap
from utils.metrics import transfer_accuracy, calc_domainAveraged_FOSCTTM

def test_params(data1, data2, label1, label2, links, m_range, e_range, k_range, save_path, acc_log_url, foscttm_log_url, save_data=True, save_fig=True):
    if not os.path.exists(acc_log_url):
        with open(acc_log_url, 'w') as f: f.write(f'Param\tltACC\n')

    if not os.path.exists(foscttm_log_url):
        with open(foscttm_log_url, 'w') as f: f.write(f'Param\tAVG_FOSCTTM\n') 

    if save_data:
        os.makedirs(os.path.join(save_path, 'mapping'), exist_ok=True)
        os.makedirs(os.path.join(save_path, 'integration'), exist_ok=True)


    for m in m_range:
        for e in e_range:
            for k in k_range:
                print("Runing mode={}, metric={}, k={}, e={}".format(m[0], m[1], k, e))
                scot_instance = scotv1.SCOT(data1, data2)
                X_aligned, y_aligned = scot_instance.align(k=k, e=e, mode=m[0], metric=m[1], normalize=False)

                integrated_data = [X_aligned, y_aligned]
                mapping = scot_instance.coupling

                # save integration result
                if save_data:
                    np.savetxt(os.path.join(save_path, "mapping", "m{}_e{}_k{}.txt".format(m[0][0], e, k)), mapping)
                    np.savetxt(os.path.join(save_path, "integration", "m{}_e{}_k{}_data1.txt".format(m[0][0], e, k)), integrated_data[0])
                    np.savetxt(os.path.join(save_path, "integration", "m{}_e{}_k{}_data2.txt".format(m[0][0], e, k)), integrated_data[1])
                if save_fig:
                    plt_heatmap(mapping, "SCOTv1_m{}_e{}_k{}".format(m[0], e, k), show=False, 
                                save_url=os.path.join(save_path, "mapping_fig", "heatmap", "m{}_e{}_k{}.png".format(m[0][0], e, k)))
                        
                # evaluate
                acc = transfer_accuracy(integrated_data[0], integrated_data[1], label1, label2)
                foscttm = calc_domainAveraged_FOSCTTM(integrated_data[0], integrated_data[1], links)

                # save acc & FOSCTTM error
                with open(acc_log_url, 'a') as f: f.write(f'm{m[0][0]}_e{e}_k{k}\t{acc}\n')    
                with open(foscttm_log_url, 'a') as f: f.write(f'm{m[0][0]}_e{e}_k{k}\t{foscttm}\n')  

def main(args):
    with open("./run_baselines/baseline.yaml", "r") as file:
        config = yaml.safe_load(file)
    params = config[args.dataset]
    assert os.path.exists(params["data_path"])

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

    # create links for FOSCTTM, all datasets follow 1-1 correspondence
    links = np.array(list(zip([i for i in range(data1.shape[0])], [i for i in range(data2.shape[0])])))

    if params["norm"]:
        data1 = normalize(data1, norm=params["norm"])
        data2 = normalize(data2, norm=params["norm"])

    if params["sort"]:
        data1, label1, sorted_indices1 = sorted_by_label(data1, label1)
        data2, label2, sorted_indices2 = sorted_by_label(data2, label2)

        # Create a mapping from original indices to sorted indices
        sorted_indices1_map = np.argsort(sorted_indices1)
        sorted_indices2_map = np.argsort(sorted_indices2)

        # Update the links array using the mapping
        updated_links = np.zeros_like(links)
        updated_links[:, 0] = sorted_indices1_map[links[:, 0]]
        updated_links[:, 1] = sorted_indices2_map[links[:, 1]]
        links = updated_links

    save_path = os.path.join(params["save_path"], 'scotv1')
    os.makedirs(os.path.join(save_path, 'metrics'), exist_ok=True)

    # create acc & FOSCTTM error files if not exists
    acc_log_url = os.path.join(save_path, 'metrics', 'accLT_log.txt')
    foscttm_log_url = os.path.join(save_path, 'metrics', 'FOSCTTM_log.txt')

    # test all suggested parameters
    m_range = [("distance", "euclidean"), ("connectivity", "correlation")]
    e_range = [1e-3, 5e-3, 1e-2, 5e-2, 1e-1]
    kmin=20
    kmax=min(data1.shape[0], data2.shape[0])//5
    k_range = range(kmin, kmax+1, params["scotv1"]["k_step"])
    
    test_params(data1, data2, label1, label2, links, m_range, e_range, k_range,
                   save_path = save_path, acc_log_url=acc_log_url, foscttm_log_url=foscttm_log_url)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, help='Dataset name')
    main(parser.parse_args())