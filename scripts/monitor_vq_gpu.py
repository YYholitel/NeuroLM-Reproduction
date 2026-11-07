"""Record per-device utilization on the actual TI-ONE training node."""
import argparse,subprocess,time
from pathlib import Path

parser=argparse.ArgumentParser(__doc__)
parser.add_argument('--output',required=True)
args=parser.parse_args()
fields='timestamp,index,uuid,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu'
with Path(args.output).open('a',buffering=1) as stream:
    if stream.tell()==0:
        stream.write('timestamp,gpu_index,gpu_uuid,gpu_util_pct,memory_util_pct,memory_used_MiB,memory_total_MiB,power_W,temperature_C\n')
    while True:
        result=subprocess.run(['nvidia-smi','--query-gpu='+fields,'--format=csv,noheader,nounits'],
                              capture_output=True,text=True,timeout=15)
        if result.returncode:
            raise RuntimeError('nvidia-smi monitoring failed with exit '+str(result.returncode))
        stream.write(result.stdout);stream.flush()
        time.sleep(5)
