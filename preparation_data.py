#!/bin/env python3
import os
import sys
import re
import utils

def GetNewUtt2DurAndSegment(max_seg, utt2dur, new_utt2dur_path, new_segment_path):
    utt2dur_str = ""
    segment_str = ""
    new_utt = []
    diffs = 0.0

    for utt, dur in utt2dur.items():
        duration = float(dur)
        seg_index = 1

        while(duration > max_seg):

            seg_utt = "{}-{}".format(utt, str(seg_index).zfill(6))
            new_utt.append(seg_utt)

            utt2dur_str += "{} {}\n".format(seg_utt, max_seg)
            segment_str += "{} {} {} {}\n".format(seg_utt,utt,
                                        max_seg*(seg_index-1),
                                        max_seg*seg_index)
            duration -= max_seg
            seg_index += 1

        seg_utt = "{}-{}".format(utt, str(seg_index).zfill(6))
        new_utt.append(seg_utt)

        utt2dur_str += "{} {}\n".format(seg_utt, duration)
        segment_str += "{} {} {} {}\n".format(seg_utt, utt, max_seg*(seg_index-1),
                                                    duration+max_seg*(seg_index-1))

        diff = float(max_seg) - duration 
        assert(diff >= 0)
        diffs += diff

    print("Max Segment:", max_seg, "All Diff:", diffs)

    # remove last "\n"
    utt2dur_str = utt2dur_str[:-1]
    segment_str = segment_str[:-1]

    with open(new_utt2dur_path, 'w') as fp:
        fp.write(utt2dur_str)

    with open(new_segment_path, 'w') as fp:
        fp.write(segment_str)

    return(new_utt)

def GetNewUttAndLang(new_utts, utt2lang_path):

    utt2lang_str = ""
    for utt in new_utts:
        #print(utt)
        lang = re.findall(r'^([^\-]+)-.*', utt)[0]
        utt2lang_str += "{} {}\n".format(utt, lang)

    utt2lang_str = utt2lang_str[:-1]

    with open(utt2lang_path, 'w') as fp:
        fp.write(utt2lang_str)

if __name__=="__main__":

    if len(sys.argv) < 3:
        raise ValueError(
            "Must 2 dataset: the path of source data and target data.")

    max_seg = 8

    sr_data = sys.argv[1]
    tr_data = sys.argv[2]

    print("Source Dataset:", sr_data)
    print("Target Dataset:", tr_data)

    os.makedirs(tr_data, exist_ok=True)

    utt2dur = utils.Read2Column(os.path.join(sr_data,"utt2dur"))
    new_utt = GetNewUtt2DurAndSegment(max_seg, utt2dur,
                            os.path.join(tr_data, "utt2dur"),
                            os.path.join(tr_data, "segments"))

    GetNewUttAndLang(new_utt, os.path.join(tr_data, "utt2lang"))

    #for max_seg in range(4, 25):
    #    GetNewUtt2DurAndSegment(max_seg, utt2dur,
    #                            os.path.join(tr_data, "utt2dur"),
    #                            os.path.join(tr_data, "segments"))

