#!/bin/bash

nj=10
steps=
cmd="slurm.pl --quiet"
python=/home/zhb502/anaconda3/bin/python
###------------------------------------------------

echo 
echo "$0 $@"
echo

set -e

. path.sh

. parse_options.sh || exit 1

steps=$(echo $steps | perl -e '$steps=<STDIN>;  $has_format = 0;
  if($steps =~ m:(\d+)\-$:g){$start = $1; $end = $start + 10; $has_format ++;}
        elsif($steps =~ m:(\d+)\-(\d+):g) { $start = $1; $end = $2; if($start == $end){}
        elsif($start < $end){ $end = $2 +1;}else{die;} $has_format ++; }
      if($has_format > 0){$steps=$start;  for($i=$start+1; $i < $end; $i++){$steps .=":$i"; }}
      print $steps;' 2>/dev/null)  || exit 1

if [ ! -z "$steps" ]; then
  for x in $(echo $steps|sed 's/[,:]/ /g'); do
  index=$(printf "%02d" $x);
  declare step$index=1
  done
fi

# CUDA_VISIBLE_DEVICES="3"

fbankdim=80
trn_set=train_seg
dev_set=dev_all
trn_dir=exp/data/${trn_set}_fbank${fbankdim}
dev_dir=exp/data/${dev_set}_fbank${fbankdim}
exp_dir=exp/fbank${fbankdim}_model9

if [ ! -z $step02 ]; then
    echo "##: Feature Generation"
    for x in $trn_set $dev_set; do
        data=exp/data/${x}_fbank${fbankdim}
        fbankdir=$data/fbank
        logdir=$fbankdir/log
        utils/copy_data_dir.sh data/$x $data || exit 1;
        make_fbank.sh --cmd "${cmd}" --nj $nj \
            --fs 16000 \
            --fmax "7600" \
            --fmin "80" \
            --n_fft 1024 \
            --n_shift 256 \
            --win_length "" \
            --n_mels $fbankdim \
            $data $logdir $fbankdir
        utils/fix_data_dir.sh $data
    done
fi

if [ ! -z $step03 ]; then
    echo "##: Train Start:"
    CUDA_VISIBLE_DEVICES="2" \
    $python langid/train.py  $trn_dir $dev_dir $exp_dir || exit 1;
fi

