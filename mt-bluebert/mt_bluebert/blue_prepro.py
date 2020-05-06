"""
Preprocessing BLUE dataset.

Usage:
    blue_prepro [options] --root_dir=<dir> --task_def=<file> --datasets=<str>

Options:
    --overwrite
"""
import os

import docopt

from mt_bluebert.data_utils.log_wrapper import create_logger
from mt_bluebert.blue_exp_def import BlueTaskDefs
from mt_bluebert.blue_utils import load_sts, load_mednli, \
    load_relation, load_ner, dump_rows


def main(args):
    root = args['--root_dir']
    assert os.path.exists(root)

    log_file = os.path.join(root, 'blue_prepro.log')
    logger = create_logger(__name__, to_disk=True, log_file=log_file)

    task_defs = BlueTaskDefs(args['--task_def'])

    canonical_data_suffix = "canonical_data"
    canonical_data_root = os.path.join(root, canonical_data_suffix)
    if not os.path.isdir(canonical_data_root):
        os.mkdir(canonical_data_root)

    if args['--datasets'] == 'all':
        tasks = task_defs.tasks
    else:
        tasks = args['--datasets'].split(',')
    for task in tasks:
        logger.info("Task %s" % task)
        if task not in task_defs.task_def_dic:
            raise KeyError('%s: Cannot process this task' % task)

        if task in ['clinicalsts', 'biosses']:
            load = load_sts
        elif task == 'mednli':
            load = load_mednli
        elif task in ('chemprot', 'i2b2-2010-re', 'ddi2013-type'):
            load = load_relation
        elif task in ('bc5cdr-disease', 'bc5cdr-chemical', 'shareclefe'):
            load = load_ner
        else:
            raise KeyError('%s: Cannot process this task' % task)

        data_format = task_defs.data_format_map[task]
        split_names = task_defs.split_names_map[task]
        for split_name in split_names:
            fin = os.path.join(root, f'{task}/{split_name}.tsv')
            fout = os.path.join(canonical_data_root, f'{task}_{split_name}.tsv')
            if os.path.exists(fout) and not args['--overwrite']:
                logger.warning('%s: Not overwrite %s: %s', task, split_name, fout)
                continue
            data = load(fin)
            logger.info('%s: Loaded %s %s samples', task, len(data), split_name)
            dump_rows(data, fout, data_format)

        logger.info('%s: Done', task)


if __name__ == '__main__':
    args = docopt.docopt(__doc__)
    main(args)
