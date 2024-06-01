$ErrorActionPreference = "Stop"

.\run_first.ps1

python parse_history.py | Tee-Object -FilePath model_history.txt
python nnet_stats.py | Tee-Object -FilePath classifier_accuracy.txt
python mcols_stdevs.py | Tee-Object -FilePath memory_performance.txt

foreach ($n in 128, 256, 512, 1024) {
    python noised_classif.py --domain=$n
    python choose.py --domain=$n
    python check_chosen.py --domain=$n
}

.\run_second.ps1

python system_stats.py | Tee-Object -FilePath mem_nsd_prtl_performance.txt
