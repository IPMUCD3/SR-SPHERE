
#!/bin/bash

basedir=/gpfs02/work/akira.tokiwa/gpgpu/Github/SR-SPHERE/run/diffusion/train

# arguments sample
# n_maps=None
# nside=512
# order=2
# batch_size=4
# difference=True
# conditioning="concat"
# norm_type="batch"
# act_type="silu"
# use_attn=False
# mask=False
# scheduler="linear"
# timesteps=2000
# device=5

# sample command
# bash train_run.sh 100 512 2 32 True concat batch silu False True linear 2000 5

n_maps=$1
nside=$2
order=$3
batch_size=$4
difference=$5
conditioning=$6
norm_type=$7
act_type=$8
use_attn=$9
mask=${10}
scheduler=${11}
timesteps=${12}
device=${13}

# create a file name
fname=n"$n_maps"_s"$nside"_o"$order"_b"$batch_size"_d"$difference"_"$conditioning"_"$norm_type"_"$act_type"_a"$use_attn"_m"$mask"_"$scheduler"_t"$timesteps"

# duplicate the template file as fname.sh
cp $basedir/train_template.sh $basedir/used/$fname.sh

# replace the template file with the values
sed -i "2s/fname/$fname/" $basedir/used/$fname.sh
sed -i "4s/fname/$fname/" $basedir/used/$fname.sh
sed -i "5s/fname/$fname/" $basedir/used/$fname.sh
sed -i "16s/_device/$device/" $basedir/used/$fname.sh
sed -i "21s/_n_maps/$n_maps/" $basedir/used/$fname.sh
sed -i "21s/_nside/$nside/" $basedir/used/$fname.sh
sed -i "21s/_order/$order/" $basedir/used/$fname.sh
sed -i "21s/_batch/$batch_size/" $basedir/used/$fname.sh
sed -i "21s/_difference/$difference/" $basedir/used/$fname.sh
sed -i "21s/_cond/$conditioning/" $basedir/used/$fname.sh
sed -i "21s/_norm/$norm_type/" $basedir/used/$fname.sh
sed -i "21s/_act/$act_type/" $basedir/used/$fname.sh
sed -i "21s/_use_attn/$use_attn/" $basedir/used/$fname.sh
sed -i "21s/_mask/$mask/" $basedir/used/$fname.sh
sed -i "21s/_scheduler/$scheduler/" $basedir/used/$fname.sh
sed -i "21s/_timesteps/$timesteps/" $basedir/used/$fname.sh
sed -i "21s/_log_name/$fname/" $basedir/used/$fname.sh

# submit the job
sbatch $basedir/used/$fname.sh