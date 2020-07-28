import os
import re
from hparams import hparams

def ReadLang2UttGetLangLabel(lang2utt_path):
    lang2utt = Read2Column(lang2utt_path)
    lang_list = list(lang2utt.keys())
    lang_list.sort()
    lang_dict = dict()
    for i in range(len(lang_list)):
        lang_dict[lang_list[i]] = i

    return lang_dict, lang_list

def Read2Column(file_path):
    the_dict = dict()
    with open(file_path, 'r') as fp:
        all_con = fp.read().splitlines()
        for lin in all_con:
            lin = lin.strip()
            pat = r'(\S+)\s+(.+)'
            tmp = re.findall(pat, lin)[0]
            co1 = tmp[0]
            co2 = tmp[1]
            if " " in co2 and '/' in co2 and 'wav' in co2:
                co2sp = co2.split()
                for it in co2sp:
                    if '/' in it and 'wav' in it:
                        co2 = it
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
    lang2lang_id = dict()
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
    #TestReadUtt2Lang()
    #ReadLang2UttGetLangLabel("data/trn/spk2utt")
    utt2wav = Read2Column('data/dev_all/wav.scp')
    for utt,wav in utt2wav.items():
        print(utt, wav)
