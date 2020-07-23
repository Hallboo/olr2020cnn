import os
import re
from hparams import hparams

def Read2Column(file_path):
    the_dict = dict()
    with open(file_path, 'r') as fp:
        all_con = fp.read().splitlines()
        for line in all_con:
            line = line.strip()
            if re.match(r'.*sox.*', line):
                raise ValueError("Wrong wav.scp")
            pat = r'(\S+)\s+(.+)'
            co1 = re.findall(pat, line)[0][0]
            co2 = re.findall(pat, line)[0][1]
            if co1 not in the_dict:
                the_dict[co1] = co2
            else:
                raise ValueError(
                    "Different records exist for one UTT in the file")
    return the_dict

def ConvertDict(target_dict):
    v1 = list(target_dict.values())
    v2 = set(v1)
    if len(v1) != len(v2):
        raise ValueError("The dictionary cannot convert.")
    
    new_dict = dict()
    for k,v in target_dict.items():
        new_dict[v] = k

    return new_dict

def TestRead2Column():
    the_dict = Read2Column("data/train/feats.scp")
    for k,v in the_dict.items():
        print("K",k,"V",v)
    the_dict = ConvertDict(the_dict)
    for k,v in the_dict.items():
        print("K",k,"V",v)
    
def ReadWavScp(wav_scp_path):
    utt2wav = Read2Column(wav_scp_path)
    wav2utt = ConvertDict(utt2wav)
    return (utt2wav, wav2utt)

def ReadFeatsScp(feats_scp_path):
    utt2feats = Read2Column(feats_scp_path)
    return utt2feats

def ReadUtt2Lang(utt2lang_path):
    utt2lang = Read2Column(utt2lang_path)
    utt2lang_id = dict()
    lang2lang_id = GetLang2LangID()
    for utt,lang in utt2lang.items():
            utt2lang_id[utt] = lang2lang_id[lang]
    return (utt2lang, utt2lang_id)

def TestReadUtt2Lang():
    utt2lang, utt2lang_id = ReadUtt2Lang("data/train/utt2lang")
    for k,v in utt2lang.items():
        print(k,v,utt2lang_id[k], hparams.lang)

def GetFeatScp(utt2wav, feats_dir):
    feats = dict()
    for utt in utt2wav.keys():
        feats[utt] = os.path.join(feats_dir, utt + ".npy")
    return feats

def GetLang2LangID():
    lang2lang_id=dict()
    for i in range(len(hparams.lang)):
        lang2lang_id[hparams.lang[i]] = i
    return lang2lang_id

if __name__ == "__main__":

    #for k,v in get_lang2lang_id().items():
    #    print(k,v)
    # 
    #for k,v in get_index2label().items():
    #    print(k,v)
    #TestRead2Column() 
    TestReadUtt2Lang()
