import os, argparse
import yaml
import time
import psutil
import sys
import matplotlib.pyplot as plt
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "src"))
import sonata
from utils.utils import load_data
from run_baselines.scot import scotv1
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "src/run_baselines"))
from run_baselines.unioncom import UnionCom

def get_memory_usage():
    """Return current and peak memory in MB."""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss / 1e6  # Convert bytes to MB

def test_sonata(params, output_path, data_path, n_cells, scalebleOT=False, scale_sample_rate=0.1):
    data = load_data(f"{data_path}/domain1_{n_cells}.txt", )

    # start checking time & memory 
    start_time = time.time()
    mem_before = get_memory_usage()
    
    if scalebleOT:
        params.update({"scalableOT": True})
    params.update({"scale_sample_rate": scale_sample_rate})
    sn = sonata.sonata(params)
    DiagnoseResult1 = sn.diagnose(data, save_dir=os.path.join(f"{output_path}_{n_cells}", "Modality1"))

    # stop time & memory
    end_time = time.time()
    mem_after = get_memory_usage()

    peak_mem = max(mem_before, mem_after)  # Optional: could also monitor periodically for peak
    
    method_name = f"scalable_SONATA{scale_sample_rate}" if scalebleOT else "SONATA"   
    with open(f"{output_path}/diagnose_SONATA_log.txt", "a") as f:
        f.write(f"{method_name}, {n_cells}, {end_time - start_time:.2f}, {peak_mem:.2f}\n")

def test_baselines(params, output_path, data_path, n_cells, method):
    data1 = load_data(f"{data_path}/domain1_{n_cells}.txt", )    
    data2 = load_data(f"{data_path}/domain2_{n_cells}.txt", ) 
    
    # start checking time & memory 
    start_time = time.time()
    mem_before = get_memory_usage()
    
    if method == "scot":
        scot = scotv1.SCOT(data1.copy(), data2.copy())
        x_aligned, y_aligned = scot.align(k = params["scot_k"], e=params["scot_e"], mode=params["scot_mode"], metric=params["scot_metric"], normalize=params["norm"])  
    elif method == "unioncom":
        uc = UnionCom.UnionCom(rho=1, kmin=5, epoch_pd=2000, epsilon=0.01, log_pd=100)
        integrated_data, mapping = uc.fit_transform(dataset=[data1,data2])
    else:
        raise ValueError(f"Method should be 'scot' or 'unioncom'")

    # stop time & memory
    end_time = time.time()
    mem_after = get_memory_usage()

    peak_mem = max(mem_before, mem_after)  # Optional: could also monitor periodically for peak
        
    with open(f"{output_path}/diagnose_baselines_log.txt", "a") as f:
        f.write(f"{method}, {n_cells}, {end_time - start_time:.2f}, {peak_mem:.2f}\n")    

def plot_scalability(fpath, output_path):
    # load data
    methods = {}
    with open(fpath, 'r') as file:
        for line in file:
            parts = line.strip().split(',')
            method = parts[0].strip()
            cell_count = int(parts[1].strip())
            time = float(parts[2].strip())
            
            # save as dict
            if method not in methods:
                methods[method] = {'cell_counts': [], 'times': []}
            methods[method]['cell_counts'].append(cell_count)
            methods[method]['times'].append(time)

    # plot figure
    plt.figure(figsize=(10, 6))
    for method, data in methods.items():
        plt.plot(data['cell_counts'], data['times'], linestyle='--', marker='o', label=method)
        
    # figure settings
    plt.title('Scalability Comparison', fontsize=16)
    plt.xlabel('Number of Cells', fontsize=14)
    plt.ylabel('Time (seconds)', fontsize=14)
    plt.legend(fontsize=12)
    plt.tight_layout()

    plt.savefig(f"{output_path}/diagnose_scalability_plot.png")
    plt.close()
      
def main(args):
    with open("./cfgs/t_branch.yaml", "r") as file:
            params = yaml.safe_load(file)
    
    # if args.method == "sonata":
    #     test_sonata(params, args.output_path, args.data_path, args.n_cells, scale_sample_rate=args.scale_sample_rate)
    # else:
    #     test_baselines(params, args.output_path, args.data_path, args.n_cells, args.method)

    plot_scalability(fpath=f"{args.output_path}/diagnose_baselines_log_all.txt", output_path=args.output_path)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sonata Scalability Test")
    parser.add_argument("--n_cells", type=int, default=300, help="Number of cells to test")
    parser.add_argument("--data_path", type=str, default="../results/scalability/t_branch", help="Output path for results")
    parser.add_argument("--output_path", type=str, default="../results/scalability/t_branch", help="Output path for results")
    parser.add_argument("--method", type=str, default="sonata", help="test method: sonata, scot or unioncom")
    parser.add_argument("--scale_sample_rate", type=float, default=0.1, help="sample rate for scalable OT")
    args = parser.parse_args()

    main(args)