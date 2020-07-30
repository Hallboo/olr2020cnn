#!/bin/bash

nj=10
steps=15
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

# step1: re segment data
if [ ! -z $step01 ]; then
    sr_data=data/train
    tr_data=data/train_seg
    
    # get utt2lang segments
    python scripts/preparation_data.py 4 $sr_data $tr_data || exit 1;
    utils/utt2spk_to_spk2utt.pl < $tr_data/utt2lang > $tr_data/lang2utt || exit 1;

    cp $tr_data/utt2lang $tr_data/utt2spk
    cp $tr_data/lang2utt $tr_data/spk2utt
    #cp $sr_data/wav.scp $tr_data
    cat $sr_data/wav.scp | perl -ane 'chomp;m/(\S+)\s+(\S+)/g;($utt,$wav)=($1,$2);
        print("$utt sox -t wav $wav -c 1 -b 16 -t wav -r 16000 - |\n");' > $tr_data/wav.scp

    utils/fix_data_dir.sh $tr_data || exit 1;
fi

# step2: prepare trn for train
#        prepare dev for validation
if [ ! -z $step02 ]; then
    #sub step1: count num of trn and dev
    tr_seg=data/train_seg
    num_utt=$(wc -l $tr_seg/utt2lang|awk '{print $1}')
    trn_utt=$(echo "print(int($num_utt*0.8))"|python)
    dev_utt=$(echo "print($num_utt-$trn_utt)"|python)
    echo "$num_utt $trn_utt $dev_utt"
    
    #sub step2: get uttlist for trn and dev
    trn_dir=data/trn
    dev_dir=data/dev
    [ -d $trn_dir ] && rm -rf $trn_dir
    [ -d $dev_dir ] && rm -rf $dev_dir
    mkdir -p $trn_dir
    mkdir -p $dev_dir

    awk '{print $1}' $tr_seg/utt2lang | shuf > $tr_seg/shufuttlist.txt || exit 1;
    cat $tr_seg/shufuttlist.txt | head -n $trn_utt > $trn_dir/uttlist.txt || exit 1;
    cat $tr_seg/shufuttlist.txt | tail -n $dev_utt > $dev_dir/uttlist.txt || exit 1;
    echo "Trn Set:" && wc -l $trn_dir/uttlist.txt
    echo "Dev Set:" && wc -l $dev_dir/uttlist.txt
    
    #sub step3: get subset for trn and dev
    utils/data/subset_data_dir.sh --utt-list $trn_dir/uttlist.txt $tr_seg $trn_dir || exit 1;
    utils/data/subset_data_dir.sh --utt-list $dev_dir/uttlist.txt $tr_seg $dev_dir || exit 1;
    utils/fix_data_dir.sh $trn_dir || exit 1;
    utils/fix_data_dir.sh $dev_dir || exit 1;
fi

trn_dir=data/trn
dev_dir=data/dev

if [ ! -z $step03 ]; then
    echo "##: Feature Generation"
    for x in $trn_dir $dev_dir; do
        data=$x
        fbankdir=$x/fbank
        logdir=$fbankdir/log
        #steps/make_fbank.sh --cmd "${cmd}" --nj $nj $data $logdir $fbankdir || exit 1;
        make_fbank.sh --cmd "${cmd}" --nj $nj \
            --fs 16000 \
            --fmax "7600" \
            --fmin "80" \
            --n_fft 1024 \
            --n_shift 256 \
            --win_length "" \
            --n_mels 40 \
            $data $logdir $fbankdir
        utils/fix_data_dir.sh $data
    done
fi

exp_dir=exp/train4second
dev_dir=data/dev_all

if [ ! -z $step04 ]; then
    echo "##: Train Start:"
    CUDA_VISIBLE_DEVICES="3" \
    $python langid/train.py  $trn_dir $dev_dir $exp_dir || exit 1;
        #--resume exp/train4second/best.pth \
        #$trn_dir $dev_dir $exp_dir || exit 1;
fi

if [ ! -z $step05 ]; then
    for x in data/dev_all; do
        data=$x
        fbankdir=$x/fbank
        logdir=$fbankdir/log
        make_fbank.sh --cmd "${cmd}" --nj $nj \
            --fs 16000 \
            --fmax "7600" \
            --fmin "80" \
            --n_fft 1024 \
            --n_shift 256 \
            --win_length "" \
            --n_mels 40 \
            $data $logdir $fbankdir
        utils/fix_data_dir.sh $data
    done

fi

if [ ! -z $step06 ]; then
    CUDA_VISIBLE_DEVICES="1" $python langid/evaluate.py data/dev_all exp/train4second
fi

if [ ! -z $step07 ]; then
    dev_all_dir=data/dev_all
    [ -d $dev_all_dir  ] || mkdir -p $dev_all_dir
    src_dir=/home/pyz504/deliverables/olr2020/baseline_data/train_temp
    grep dev_all /home/pyz504/deliverables/olr2020/baseline_data/train_temp/wav.scp > $dev_all_dir/uttlist.txt
    utils/subset_data_dir.sh --utt-list $dev_all_dir/uttlist.txt $src_dir $dev_all_dir
    cp $dev_all_dir/wav.scp $dev_all_dir/wav_tmp.scp
    cat $dev_all_dir/wav_tmp.scp | perl -ane 'chomp;m/(\S+)\s+(\S+)/g;($utt,$wav)=($1,$2);print("$utt sox -t wav $wav -c 1 -b 16 -t wav -r 16000 - |\n");' > \
        $dev_all_dir/wav.scp
    rm -rf $dev_all_dir/wav_tmp.scp
    ls $dev_all_dir
fi

if [ ! -z $step08 ]; then
    dataset=data/dev_all
    cat $dataset/utt2lang | perl -ane 'chomp;m/(\S+)\s+(\S+)/g;($utt,$lang)=($1,$2);$new_lang=$lang;
    $new_lang=~s/([a-zA-Z]+)-([a-zA-Z]+)/\1_\2/g;if($new_lang=~m/^$/g){$new_lang=$lang;}
    $new_utt=$utt;$new_utt=~s/^$lang/$new_lang/g;
    print("$new_utt $new_lang\n")' > $dataset/utt2spk || exit 1;
    sort $dataset/utt2spk -o $dataset/utt2spk
    sort $dataset/wav.scp -o $dataset/wav.scp
    awk '{print $1}' $dataset/utt2spk | paste - $dataset/wav.scp > $dataset/wavtmp
    awk '{$2="";print}' $dataset/wavtmp > $dataset/wav.scp
    cp $dataset/utt2spk $dataset/utt2lang || exit 1;
    utils/utt2spk_to_spk2utt.pl < $dataset/utt2lang > $dataset/lang2utt || exit 1;
    utils/utt2spk_to_spk2utt.pl < $dataset/utt2spk  > $dataset/spk2utt  || exit 1;
    utils/fix_data_dir.sh $dataset
fi
#all_dataset="task1_dev  task1_dev_enroll  task1_enroll  task2_dev  task2_enroll  task3_enroll  train"
