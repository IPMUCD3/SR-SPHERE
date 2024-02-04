
basedir=/gpfs02/work/akira.tokiwa/gpgpu/Github/SR-SPHERE/run/diffusion/validation

# arguments sample
# target="HR"
# model="diffusion"
# transform_type="sigmoid"
# cond="concat"
# schedule="linear"
# ifmask=True
# order=8
# nmap=100
# batch=24
# device=5

# sample command
# bash valid_run.sh HR diffusion sigmoid concat linear True 8 100 24 5

target=$1
model=$2
transform_type=$3
cond=$4
schedule=$5
ifmask=$6
order=$7
nmap=$8
batch=$9
device=${10}

# create a file name
fname=valid_"$model"_"$target"_"$transform_type"_"$cond"_"$schedule"_"$ifmask"_"$order"_"$nmap"_"$batch"

# duplicate the template file as fname.sh
cp $basedir/valid_template.sh $basedir/used/$fname.sh

# replace the template file with the values
sed -i "2s/fname/$fname/" $basedir/used/$fname.sh
sed -i "4s/fname/$fname/" $basedir/used/$fname.sh
sed -i "5s/fname/$fname/" $basedir/used/$fname.sh
sed -i "16s/_device/$device/" $basedir/used/$fname.sh
sed -i "21s/_model/$model/" $basedir/used/$fname.sh
sed -i "21s/_target/$target/" $basedir/used/$fname.sh
sed -i "21s/_ifmask/$ifmask/" $basedir/used/$fname.sh
sed -i "21s/_schedule/$schedule/" $basedir/used/$fname.sh
sed -i "21s/_trans/$transform_type/" $basedir/used/$fname.sh
sed -i "21s/_cond/$cond/" $basedir/used/$fname.sh
sed -i "21s/_order/$order/" $basedir/used/$fname.sh
sed -i "21s/_nmap/$nmap/" $basedir/used/$fname.sh
sed -i "21s/_batch/$batch/" $basedir/used/$fname.sh

# submit the job
sbatch $basedir/used/$fname.sh

# delete the file
# rm $basedir/used/$fname.sh