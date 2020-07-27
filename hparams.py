from src.tfcompat.hparam import HParams
import numpy as np

# Default hyperparameters:
hparams = HParams(
    name="olr",
    sample_rate=16000,
    num_mels=40,
    n_fft=int(0.04*16000),
    # n_fft=2048,
    hop_length=int(0.02*16000),
    win_length=int(0.04*16000),
    deltas=False,

    # training testing evaluating
    model_type='Cnn_9layers_AvgPooling',

    use_cuda=False,
    max_epoch=100,
    batch_size=128,
    
    lang=None
        #'Kazak',
        #'TE_IN',
        #'Tibet',
        #'Uyghu',
        #'ca_es',
        #'ct_cn',
        #'el_gr',
        #'id_id',
        #'ja_jp',
        ##'ko_kr',
        #'ru_ru',
        #'shanghai',
        #'sichuan',
        #'minnan',
        #'vi_vn',
        #'zh_cn',
    #],
)

def hparams_debug_string():
    values = hparams.values()
    hp = ['  %s: %s' % (name, values[name]) for name in sorted(values)]
    return 'Hyperparameters:\n' + '\n'.join(hp)
