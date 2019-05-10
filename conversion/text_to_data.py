import os
from multiprocessing import Pool
import glob
import re
import argparse
from collections import OrderedDict

patterns = OrderedDict()
patterns['newlines'] = (re.compile(r'[\n\r]+'), ' ')
patterns['alphanum_word'] = (re.compile(r'\S*\d+\S*'), ' N ') # Anything with a number
patterns['eos_puncts'] = (re.compile(r"[.!]+"), ' | ') # End of Sentence Puctuations without Question Mark
patterns['q_mark'] = (re.compile(r"\?+"), ' ? ')
patterns["non_dict_chars"] = (re.compile(r'[^a-zN|?\n ]'), '')
patterns["multi_space"] = (re.compile(r'[ ]{2,}'), ' ')
patterns["start_space"] = (re.compile(r'\n\s+'), '\n')


def clean_file(input_file, output_file, clean_type):
    if clean_type=='end_sent':
        patterns['eos_puncts'] = (re.compile(r"[.!]+"), ' |\n') # End of Sentence Puctuations without Question Mark
        patterns['q_mark'] = (re.compile(r"\?+"), ' ?\n')

    text = None
    with open(input_file, 'r') as inp:
        text = inp.read().lower().strip()
        for k, p in patterns.items():
            text = p[0].sub(p[1], text)

    if clean_type=='mid_sent':
        pass #TODO

    with open(output_file, 'w') as out:
        out.write(text)


if __name__=="__main__":

    global args
    parser = argparse.ArgumentParser(description='PreProcessing the raw files')
    parser.add_argument('--input_dir', type=str, required=True, help='Directory containing files to be processed')
    parser.add_argument('--output_dir', type=str, required=True, help='Destination Directory')
    parser.add_argument('--clean_type', type=str, choices=['end_sent', 'mid_sent'], required=True, help='What kind of processing do you want.')

    args = parser.parse_args()


    try:
        os.makedirs(args.output_dir)
    except OSError:
        check = input("'output_dir' already exists. Are you sure you want to continue? (y/n)")
        if check.lower() in ['n', 'no', 'no.']:
            exit(1)
    
    input_files = glob.glob(os.path.join(args.input_dir, "**"), recursive=True)
    data_to_process = []
    for input_file in input_files:
        if not os.path.isdir(input_file):
            file_name = os.path.basename(input_file)
            output_file = os.path.join(args.output_dir, file_name)
            data_to_process.append((input_file, output_file, args.clean_type))

    with Pool() as pool:
        pool.starmap(clean_file, data_to_process)


