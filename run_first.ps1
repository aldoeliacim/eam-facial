$ErrorActionPreference = "Stop"

foreach ($n in 128, 256, 512, 1024) {
    $dir = "runs-$n"
    if (-Not (Test-Path $dir)) {
        Write-Output "$dir not found, creating and copying default mem_params.csv"
        New-Item -ItemType Directory -Path $dir
        Copy-Item -Path "mem_params.csv" -Destination "$dir"
    }

    python eam.py -n --domain=$n --runpath=$dir
    python eam.py -f --domain=$n --runpath=$dir
    
    Write-Output "Running experiment with domain size $n"
    python eam.py -e 1 --domain=$n --runpath=$dir

    if ($LASTEXITCODE -ne 0) {
        Write-Output "Failed at domain size $n"
        break
    }
}
