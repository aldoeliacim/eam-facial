$ErrorActionPreference = "Stop"

foreach ($n in 128, 256, 512, 1024) {
    $dir = "runs-$n"
    python eam.py -r --domain=$n --runpath=$dir
    python eam.py -d --domain=$n --runpath=$dir

    if ($LASTEXITCODE -ne 0) {
        Write-Output "Failed at domain size $n"
        break
    }
}
